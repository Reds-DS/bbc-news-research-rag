import asyncio
import csv
import json
import logging
import math
import os
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
    question: str
    answer: str
    contexts: list[str]
    faithfulness: float = float("nan")
    response_relevancy: float = float("nan")


@dataclass
class EvaluationReport:
    results: list[QuestionResult] = field(default_factory=list)
    avg_faithfulness: float = float("nan")
    avg_response_relevancy: float = float("nan")


def save_evaluation_log(report: EvaluationReport) -> tuple[str, str]:
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


class RAGASEvaluator:
    def __init__(self):
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
        return asyncio.run(self.aevaluate(results))


def load_questions(file_path: str) -> list[str]:
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
) -> list[QuestionResult]:
    retriever = Retriever()
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
) -> list[QuestionResult]:
    return asyncio.run(async_run_rag_pipeline(
        questions,
        context_limit=context_limit,
        progress_callback=progress_callback,
    ))


async def async_evaluate_results(
    results: list[QuestionResult],
    max_concurrency: int = 5,
) -> EvaluationReport:
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
    return asyncio.run(async_evaluate_results(results))


async def async_run_evaluation(
    file_path: str,
    context_limit: int = 5,
    ollama_concurrency: int = 3,
    eval_concurrency: int = 5,
    progress_callback=None,
) -> EvaluationReport:
    questions = load_questions(file_path)
    results = await async_run_rag_pipeline(
        questions,
        context_limit=context_limit,
        ollama_concurrency=ollama_concurrency,
        progress_callback=progress_callback,
    )
    report = await async_evaluate_results(results, max_concurrency=eval_concurrency)
    return report


def run_evaluation(
    file_path: str,
    context_limit: int = 5,
    progress_callback=None,
) -> EvaluationReport:
    return asyncio.run(async_run_evaluation(
        file_path,
        context_limit=context_limit,
        progress_callback=progress_callback,
    ))
