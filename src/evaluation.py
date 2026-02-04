import asyncio
import json
import logging
import math
from dataclasses import dataclass, field

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.metrics import Faithfulness, AnswerRelevancy
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

    def evaluate(self, results: list[QuestionResult]) -> list[dict]:
        scores_list = []
        for result in results:
            scores = {}
            try:
                faith_result = self.faithfulness.ascore(
                    user_input=result.question,
                    response=result.answer,
                    retrieved_contexts=result.contexts,
                )
                if asyncio.iscoroutine(faith_result):
                    faith_result = asyncio.get_event_loop().run_until_complete(faith_result)
                scores["faithfulness"] = faith_result.value
            except Exception as e:
                logger.error(f"Faithfulness scoring failed: {type(e).__name__}: {e}")
                scores["faithfulness"] = float("nan")

            try:
                rr_result = self.response_relevancy.ascore(
                    user_input=result.question,
                    response=result.answer,
                )
                if asyncio.iscoroutine(rr_result):
                    rr_result = asyncio.get_event_loop().run_until_complete(rr_result)
                scores["response_relevancy"] = rr_result.value
            except Exception as e:
                logger.error(f"Response relevancy scoring failed: {type(e).__name__}: {e}")
                scores["response_relevancy"] = float("nan")

            scores_list.append(scores)
        return scores_list


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


def run_rag_pipeline(
    questions: list[str],
    context_limit: int = 5,
    progress_callback=None,
) -> list[QuestionResult]:
    retriever = Retriever()
    generator = RAGGenerator()
    results = []

    for i, question in enumerate(questions):
        if progress_callback:
            progress_callback(i, len(questions), question)

        context = retriever.search(question, limit=context_limit)
        answer = generator.answer(question, context)

        context_strings = [
            f"{article['title']}\n{article['description']}"
            for article in context
        ]

        results.append(QuestionResult(
            question=question,
            answer=answer,
            contexts=context_strings,
        ))

    return results


def evaluate_results(results: list[QuestionResult]) -> EvaluationReport:
    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(results)

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

def run_evaluation(
    file_path: str,
    context_limit: int = 5,
    progress_callback=None,
) -> EvaluationReport:
    questions = load_questions(file_path)
    results = run_rag_pipeline(questions, context_limit, progress_callback)
    report = evaluate_results(results)
    return report