import asyncio
import csv
import json
import logging
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

from ragas.llms import llm_factory
from langchain_openai import OpenAIEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample
from openai import AsyncOpenAI

from src.config import OPENAI_API_KEY, EVAL_LLM_MODEL, EVAL_EMBEDDING_MODEL
from src.retrieval import Retriever
from src.generation import RAGGenerator

logger = logging.getLogger(__name__)


@dataclass
class QuestionResult:
    """Holds a single question's RAG output and evaluation scores."""

    question: str
    answer: str
    contexts: list[str]
    faithfulness: float = float("nan")
    response_relevancy: float = float("nan")


@dataclass
class EvaluationReport:
    """Aggregated evaluation report containing per-question results and averages."""

    results: list[QuestionResult] = field(default_factory=list)
    avg_faithfulness: float = float("nan")
    avg_response_relevancy: float = float("nan")


def save_evaluation_log(report: EvaluationReport) -> tuple[str, str]:
    """Save the evaluation report to timestamped JSON and CSV files in eval_logs/.

    Creates two files with matching timestamps:
    - JSON: Full structured report with averages and per-question details.
    - CSV: Flat table for easy spreadsheet analysis.

    NaN scores are serialized as null (JSON) or empty string (CSV).

    Args:
        report: The completed EvaluationReport with scored results.

    Returns:
        tuple[str, str]: Paths to the saved (json_path, csv_path) files.
    """
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.strftime("eval_%Y-%m-%d_%H-%M-%S")
    json_path = os.path.join(log_dir, f"{timestamp_str}.json")
    csv_path = os.path.join(log_dir, f"{timestamp_str}.csv")

    def clean(value: float):
        return None if math.isnan(value) else value

    log_data = {
        "timestamp": now.isoformat(timespec="seconds"),
        "num_questions": len(report.results),
        "avg_faithfulness": clean(report.avg_faithfulness),
        "avg_response_relevancy": clean(report.avg_response_relevancy),
        "results": [
            {
                "question": r.question,
                "answer": r.answer,
                "contexts": r.contexts,
                "faithfulness": clean(r.faithfulness),
                "response_relevancy": clean(r.response_relevancy),
            }
            for r in report.results
        ],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "contexts", "faithfulness", "response_relevancy"])
        for r in report.results:
            writer.writerow([
                r.question,
                r.answer,
                "\n---\n".join(r.contexts),
                clean(r.faithfulness) or "",
                clean(r.response_relevancy) or "",
            ])

    return json_path, csv_path


def append_changelog(
    report: EvaluationReport,
    search_mode: str = "hybrid",
    notes: str = "",
) -> str:
    """Append a summary row to eval_logs/CHANGELOG.md for tracking experiments.

    Writes a Markdown table row containing timestamp, average scores, search
    mode, current git commit hash + message, and user-supplied notes. If
    the changelog file does not exist yet, creates it with a header row.

    Args:
        report: The completed EvaluationReport with average scores.
        search_mode: The retrieval mode used (e.g. 'hybrid', 'semantic').
        notes: Free-text description of what changed in this run.

    Returns:
        str: Path to the CHANGELOG.md file.
    """
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)
    changelog_path = os.path.join(log_dir, "CHANGELOG.md")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def fmt_score(value: float) -> str:
        return f"{value:.4f}" if not math.isnan(value) else "N/A"

    # Get git info
    commit_info = ""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--pretty=format:%h %s"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            commit_info = f"`{result.stdout.strip()}`"
    except Exception:
        pass

    row = (
        f"| {now} "
        f"| {fmt_score(report.avg_faithfulness)} "
        f"| {fmt_score(report.avg_response_relevancy)} "
        f"| {search_mode} "
        f"| {commit_info} "
        f"| {notes} |\n"
    )

    if not os.path.exists(changelog_path):
        header = (
            "# Evaluation Changelog\n\n"
            "| Date | Faithfulness | Resp. Relevancy | Mode | Commit | Notes |\n"
            "|------|-------------|----------------|------|--------|-------|\n"
        )
        with open(changelog_path, "w", encoding="utf-8") as f:
            f.write(header)

    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(row)

    return changelog_path


class RAGASEvaluator:
    """Evaluator that scores RAG outputs using RAGAS Faithfulness and Response Relevancy."""

    def __init__(self):
        """Initialize the RAGAS evaluator with OpenAI-backed LLM and embeddings.

        Creates an AsyncOpenAI client and uses ragas.llms.llm_factory to build
        the evaluator LLM (model from EVAL_LLM_MODEL config). Also creates a
        LangChain OpenAIEmbeddings instance for the AnswerRelevancy metric,
        which requires embed_query/embed_documents methods (RAGAS's own
        OpenAIEmbeddings does not provide these).

        Requires OPENAI_API_KEY to be set in the environment or .env file.
        """
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
        )

        evaluator_llm = llm_factory(
            model=EVAL_LLM_MODEL,
            provider="openai",
            client=client,
            temperature=0,
        )
        evaluator_embeddings = OpenAIEmbeddings(
            model=EVAL_EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )

        self.faithfulness = Faithfulness(llm=evaluator_llm)
        self.response_relevancy = AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        )

    async def _safe_faithfulness(self, result: QuestionResult) -> float:
        """Score faithfulness for a single RAG result, returning NaN on failure.

        Measures how well the generated answer is supported by the retrieved
        contexts. Wraps the RAGAS single_turn_ascore call in a try/except
        to ensure one failed question doesn't abort the entire evaluation.

        Args:
            result: A QuestionResult containing the question, answer, and
                retrieved context strings.

        Returns:
            float: Faithfulness score in [0, 1], or NaN if scoring failed.
        """
        try:
            sample = SingleTurnSample(
                user_input=result.question,
                response=result.answer,
                retrieved_contexts=result.contexts,
            )
            score = await self.faithfulness.single_turn_ascore(sample)
            return score
        except Exception as e:
            logger.error(f"Faithfulness scoring failed: {type(e).__name__}: {e}")
            return float("nan")

    async def _safe_response_relevancy(self, result: QuestionResult) -> float:
        """Score response relevancy for a single RAG result, returning NaN on failure.

        Measures how relevant the generated answer is to the original question
        using embedding-based similarity. Wraps the RAGAS single_turn_ascore
        call in a try/except for resilience.

        Args:
            result: A QuestionResult containing the question, answer, and
                retrieved context strings.

        Returns:
            float: Response relevancy score in [0, 1], or NaN if scoring failed.
        """
        try:
            sample = SingleTurnSample(
                user_input=result.question,
                response=result.answer,
                retrieved_contexts=result.contexts,
            )
            score = await self.response_relevancy.single_turn_ascore(sample)
            return score
        except Exception as e:
            logger.error(f"Response relevancy scoring failed: {type(e).__name__}: {e}")
            return float("nan")

    async def aevaluate(
        self,
        results: list[QuestionResult],
        max_concurrency: int = 5,
    ) -> list[dict]:
        """Evaluate all RAG results concurrently with semaphore-limited parallelism.

        For each result, faithfulness and response relevancy are scored in
        parallel. A semaphore limits the number of concurrent OpenAI API
        calls to avoid rate limiting.

        Args:
            results: List of QuestionResult objects to evaluate.
            max_concurrency: Maximum number of results being scored
                simultaneously (default 5).

        Returns:
            list[dict]: One dict per result with keys 'faithfulness' and
                'response_relevancy', each a float score or NaN.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _score_one(result: QuestionResult) -> dict:
            async with semaphore:
                faith, relevancy = await asyncio.gather(
                    self._safe_faithfulness(result),
                    self._safe_response_relevancy(result),
                )
            return {"faithfulness": faith, "response_relevancy": relevancy}

        return await asyncio.gather(*[_score_one(r) for r in results])

    def evaluate(self, results: list[QuestionResult]) -> list[dict]:
        """Synchronous wrapper around aevaluate().

        Creates a new event loop via asyncio.run(). Should only be called
        when no event loop is already running.

        Args:
            results: List of QuestionResult objects to evaluate.

        Returns:
            list[dict]: Same as aevaluate() — per-result score dicts.
        """
        return asyncio.run(self.aevaluate(results))


def load_questions(file_path: str) -> list[str]:
    """Load and validate a list of evaluation question strings from a JSON file.

    The file must contain a JSON array of non-empty strings. Raises on
    invalid format or empty lists to fail fast before running the pipeline.

    Args:
        file_path: Path to the JSON file (e.g. 'eval_questions.json').

    Returns:
        list[str]: Validated list of question strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the content is not a list, contains non-strings,
            or is empty.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list):
        raise ValueError("JSON file must contain a list of question strings")
    if not all(isinstance(q, str) for q in questions):
        raise ValueError("All items in the JSON list must be strings")
    if len(questions) == 0:
        raise ValueError("Questions list is empty")

    return questions


async def async_run_rag_pipeline(
    questions: list[str],
    context_limit: int = 5,
    ollama_concurrency: int = 3,
    progress_callback=None,
    search_mode: str = "hybrid",
    reranker_model: str = "none",
    beta: float | None = None,
) -> list[QuestionResult]:
    """Run the full RAG pipeline (retrieve + generate) for all questions concurrently.

    Creates a Retriever (with optional reranker) and RAGGenerator, then
    processes all questions in parallel. Ollama generation calls are
    throttled by a semaphore to respect server concurrency limits.

    Args:
        questions: List of natural-language question strings.
        context_limit: Number of articles to retrieve per question (default 5).
        ollama_concurrency: Maximum concurrent Ollama generation requests
            (default 3).
        progress_callback: Optional callable invoked after each question is
            processed, with signature (index, total, question_text).
        search_mode: Retrieval strategy — 'semantic', 'keyword', or 'hybrid'.
        reranker_model: Reranker alias ('none', 'light', 'heavy') or full
            HuggingFace model ID. 'none' disables reranking.

    Returns:
        list[QuestionResult]: One QuestionResult per question, containing the
            question, generated answer, and context strings (scores are NaN
            until evaluation).
    """
    from src.reranker import Reranker

    reranker = Reranker(reranker_model) if reranker_model != "none" else None
    retriever = Retriever(mode=search_mode, reranker=reranker, beta=beta)
    generator = RAGGenerator()
    semaphore = asyncio.Semaphore(ollama_concurrency)

    async def _process_question(i: int, question: str) -> QuestionResult:
        context = await retriever.asearch(question, limit=context_limit)

        async with semaphore:
            answer = await generator.aanswer(question, context)

        if progress_callback:
            progress_callback(i, len(questions), question)

        context_strings = [
            f"{article['title']}\n{article['description']}"
            for article in context
        ]

        return QuestionResult(
            question=question,
            answer=answer,
            contexts=context_strings,
        )

    tasks = [_process_question(i, q) for i, q in enumerate(questions)]
    return list(await asyncio.gather(*tasks))


def run_rag_pipeline(
    questions: list[str],
    context_limit: int = 5,
    progress_callback=None,
    search_mode: str = "hybrid",
    reranker_model: str = "none",
) -> list[QuestionResult]:
    """Synchronous wrapper around async_run_rag_pipeline().

    Creates a new event loop via asyncio.run(). See async_run_rag_pipeline()
    for full parameter documentation.

    Args:
        questions: List of natural-language question strings.
        context_limit: Number of articles to retrieve per question.
        progress_callback: Optional progress reporting callable.
        search_mode: Retrieval strategy.
        reranker_model: Reranker alias or 'none'.

    Returns:
        list[QuestionResult]: Results with answers and contexts (unscored).
    """
    return asyncio.run(async_run_rag_pipeline(
        questions,
        context_limit=context_limit,
        progress_callback=progress_callback,
        search_mode=search_mode,
        reranker_model=reranker_model,
    ))


async def async_evaluate_results(
    results: list[QuestionResult],
    max_concurrency: int = 5,
) -> EvaluationReport:
    """Score all RAG results with RAGAS metrics and return an EvaluationReport.

    Instantiates a RAGASEvaluator and runs faithfulness + response relevancy
    scoring on every result. Computes average scores (ignoring NaN failures)
    and packages everything into an EvaluationReport.

    Args:
        results: List of QuestionResult objects from the RAG pipeline.
        max_concurrency: Maximum number of concurrent RAGAS scoring
            tasks (default 5).

    Returns:
        EvaluationReport: Contains per-question results with scores populated,
            plus avg_faithfulness and avg_response_relevancy.
    """
    evaluator = RAGASEvaluator()
    scores = await evaluator.aevaluate(results, max_concurrency=max_concurrency)

    for result, score in zip(results, scores):
        result.faithfulness = score.get("faithfulness", float("nan"))
        result.response_relevancy = score.get("response_relevancy", float("nan"))

    faith_scores = [r.faithfulness for r in results if not math.isnan(r.faithfulness)]
    rr_scores = [r.response_relevancy for r in results if not math.isnan(r.response_relevancy)]

    report = EvaluationReport(
        results=results,
        avg_faithfulness=sum(faith_scores) / len(faith_scores) if faith_scores else float("nan"),
        avg_response_relevancy=sum(rr_scores) / len(rr_scores) if rr_scores else float("nan"),
    )

    return report


def evaluate_results(results: list[QuestionResult]) -> EvaluationReport:
    """Synchronous wrapper around async_evaluate_results().

    Args:
        results: List of QuestionResult objects to score.

    Returns:
        EvaluationReport: Scored results with averages.
    """
    return asyncio.run(async_evaluate_results(results))


async def async_run_evaluation(
    file_path: str,
    context_limit: int = 5,
    ollama_concurrency: int = 3,
    eval_concurrency: int = 5,
    progress_callback=None,
    search_mode: str = "hybrid",
    reranker_model: str = "none",
) -> EvaluationReport:
    """Run the full evaluation pipeline end-to-end: load questions, RAG, then RAGAS scoring.

    Convenience function that chains load_questions → async_run_rag_pipeline →
    async_evaluate_results into a single awaitable call.

    Args:
        file_path: Path to the JSON file containing evaluation questions.
        context_limit: Number of articles to retrieve per question.
        ollama_concurrency: Max concurrent Ollama generation requests.
        eval_concurrency: Max concurrent RAGAS evaluation requests.
        progress_callback: Optional progress reporting callable.
        search_mode: Retrieval strategy.
        reranker_model: Reranker alias or 'none'.

    Returns:
        EvaluationReport: Fully scored report with per-question results and averages.
    """
    questions = load_questions(file_path)
    results = await async_run_rag_pipeline(
        questions,
        context_limit=context_limit,
        ollama_concurrency=ollama_concurrency,
        progress_callback=progress_callback,
        search_mode=search_mode,
        reranker_model=reranker_model,
    )
    report = await async_evaluate_results(results, max_concurrency=eval_concurrency)
    return report


def run_evaluation(
    file_path: str,
    context_limit: int = 5,
    progress_callback=None,
    search_mode: str = "hybrid",
    reranker_model: str = "none",
) -> EvaluationReport:
    """Synchronous wrapper around async_run_evaluation().

    See async_run_evaluation() for full parameter documentation.

    Args:
        file_path: Path to the JSON file containing evaluation questions.
        context_limit: Number of articles to retrieve per question.
        progress_callback: Optional progress reporting callable.
        search_mode: Retrieval strategy.
        reranker_model: Reranker alias or 'none'.

    Returns:
        EvaluationReport: Fully scored report with per-question results and averages.
    """
    return asyncio.run(async_run_evaluation(
        file_path,
        context_limit=context_limit,
        progress_callback=progress_callback,
        search_mode=search_mode,
        reranker_model=reranker_model,
    ))
