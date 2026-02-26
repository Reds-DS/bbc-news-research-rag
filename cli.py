import asyncio
import json
import math
from datetime import datetime

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="BBC News RAG System")
console = Console()


@app.command()
def status():
    """Check Chroma and Ollama connection status.

    Displays a Rich table showing connectivity and details for each backend
    service: ChromaDB (collection name, document count) and Ollama (available
    models). Useful for verifying the environment is properly configured
    before running other commands.
    """
    from src.database import ChromaDB
    from src.generation import OllamaClient
    from src.config import COLLECTION_NAME

    table = Table(title="System Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")

    # Check Chroma
    try:
        db = ChromaDB()
        if db.collection_exists():
            count = db.get_count()
            table.add_row("Chroma", "Connected", f"Collection '{COLLECTION_NAME}' has {count} objects")
        else:
            table.add_row("Chroma", "Connected", f"Collection '{COLLECTION_NAME}' is empty")
    except Exception as e:
        table.add_row("Chroma", "Error", str(e))

    # Check Ollama
    client = OllamaClient()
    if client.is_available():
        models = client.list_models()
        if models:
            table.add_row("Ollama", "Connected", f"Models: {', '.join(models)}")
        else:
            table.add_row("Ollama", "Connected", "No models loaded")
    else:
        table.add_row("Ollama", "Error", "Not available")

    console.print(table)


@app.command()
def ingest(
    recreate: bool = typer.Option(False, "--recreate", "-r", help="Delete and recreate the collection"),
    csv_path: str = typer.Option(None, "--csv", "-c", help="Path to CSV file"),
):
    """Load BBC News data from CSV into ChromaDB.

    Reads articles, embeds them using the configured HuggingFace model, and
    stores them in the ChromaDB collection. Use --recreate to drop and rebuild
    the collection from scratch.

    Args:
        recreate: If True, delete the existing collection before ingesting.
        csv_path: Override path to the CSV file (defaults to config DATA_PATH).
    """
    from src.ingestion import ingest as run_ingest
    from src.config import DATA_PATH

    path = csv_path or DATA_PATH
    console.print(f"[bold]Starting ingestion from {path}[/bold]")
    run_ingest(csv_path=path, recreate=recreate)
    console.print("[green]Done![/green]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(5, "--limit", "-n", help="Number of results"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: semantic, keyword, or hybrid"),
    reranker: str = typer.Option("none", "--reranker", help="Reranker model: none, light, or heavy"),
    beta: float = typer.Option(None, "--beta", "-b", help="Hybrid search beta (0.0=keyword, 1.0=semantic); default uses config value (0.7)"),
):
    """Search for news articles using the specified retrieval mode.

    Retrieves and displays matching articles as Rich panels showing title,
    description snippet, date, and relevance score. Supports semantic,
    keyword, and hybrid search modes with optional cross-encoder reranking.

    Args:
        query: The search query string.
        limit: Number of results to return.
        mode: Search mode — 'semantic', 'keyword', or 'hybrid'.
        reranker: Reranker model — 'none', 'light', or 'heavy'.
    """
    from src.retrieval import Retriever
    from src.reranker import Reranker

    beta_str = f" | Beta: {beta}" if mode == "hybrid" and beta is not None else ""
    console.print(f"[dim]Mode: {mode}{beta_str} | Reranker: {reranker}[/dim]\n")
    console.print(f"[bold]Searching for:[/bold] {query}\n")

    reranker_obj = Reranker(reranker) if reranker != "none" else None
    retriever = Retriever(mode=mode, reranker=reranker_obj, beta=beta)
    results = retriever.search(query, limit=limit)

    for i, article in enumerate(results, 1):
        panel = Panel(
            f"[bold]{article['title']}[/bold]\n\n"
            f"{article['description'][:300]}{'...' if len(article['description']) > 300 else ''}\n\n"
            f"[dim]Date: {article['pubDate']}[/dim]\n"
            f"[dim]Score: {article['distance']:.4f}[/dim]",
            title=f"Result {i}",
            border_style="blue",
        )
        console.print(panel)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    limit: int = typer.Option(5, "--context", "-c", help="Number of articles for context"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: semantic, keyword, or hybrid"),
    reranker: str = typer.Option("none", "--reranker", help="Reranker model: none, light, or heavy"),
    beta: float = typer.Option(None, "--beta", "-b", help="Hybrid search beta (0.0=keyword, 1.0=semantic); default uses config value (0.7)"),
):
    """Ask a question using the full RAG pipeline (retrieve + generate).

    Retrieves relevant articles from ChromaDB, optionally reranks them with
    a cross-encoder, then generates an answer using the Ollama LLM grounded
    in the retrieved context. Displays the answer and source articles.

    Args:
        question: The natural-language question to answer.
        limit: Number of context articles to retrieve.
        mode: Search mode — 'semantic', 'keyword', or 'hybrid'.
        reranker: Reranker model — 'none', 'light', or 'heavy'.
    """
    from src.retrieval import Retriever
    from src.generation import RAGGenerator
    from src.reranker import Reranker

    beta_str = f" | Beta: {beta}" if mode == "hybrid" and beta is not None else ""
    console.print(f"[dim]Mode: {mode}{beta_str} | Reranker: {reranker}[/dim]\n")
    console.print(f"[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Searching for relevant articles..."):
        reranker_obj = Reranker(reranker) if reranker != "none" else None
        retriever = Retriever(mode=mode, reranker=reranker_obj, beta=beta)
        context = retriever.search(question, limit=limit)

    console.print(f"[dim]Found {len(context)} relevant articles[/dim]\n")

    with console.status("[bold green]Generating answer..."):
        generator = RAGGenerator()
        answer = generator.answer(question, context)

    console.print(Panel(answer, title="Answer", border_style="green"))

    # Show sources
    console.print("\n[bold]Sources:[/bold]")
    for article in context:
        console.print(f"  - {article['title']} ({article['pubDate']})")


@app.command()
def trace(
    question: str = typer.Argument(..., help="Question to trace through the RAG pipeline"),
    limit: int = typer.Option(5, "--context", "-c", help="Number of articles for context"),
    fetch: int = typer.Option(None, "--fetch", "-f", help="Pre-rerank candidate count (overrides context × multiplier)"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: semantic, keyword, or hybrid"),
    reranker: str = typer.Option("light", "--reranker", help="Reranker model: none, light, or heavy"),
    beta: float = typer.Option(None, "--beta", "-b", help="Hybrid search beta (0.0=keyword, 1.0=semantic); default uses config value (0.7)"),
):
    """Run the full RAG pipeline and save a detailed trace to traces/."""
    import os
    from src.retrieval import Retriever
    from src.generation import RAGGenerator
    from src.reranker import Reranker
    from src.config import HYBRID_BETA

    beta_str = f" | Beta: {beta}" if mode == "hybrid" and beta is not None else ""
    console.print(f"[dim]Mode: {mode}{beta_str} | Reranker: {reranker}[/dim]\n")
    console.print(f"[bold]Tracing:[/bold] {question}\n")

    reranker_obj = Reranker(reranker) if reranker != "none" else None
    retriever = Retriever(mode=mode, reranker=reranker_obj, beta=beta)

    with console.status("[bold green]Retrieving and reranking articles..."):
        pre_rerank, post_rerank = retriever.trace_search(question, limit=limit, fetch_limit=fetch)

    console.print(f"[dim]Pre-rerank candidates: {len(pre_rerank)} | Post-rerank results: {len(post_rerank)}[/dim]\n")

    generator = RAGGenerator()
    with console.status("[bold green]Generating answer..."):
        prompt, answer = generator.trace_answer(question, post_rerank)

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    trace_data = {
        "question": question,
        "timestamp": now.isoformat(timespec="seconds"),
        "search_mode": mode,
        "beta": beta if beta is not None else HYBRID_BETA,
        "reranker": reranker,
        "pre_rerank_results": [
            {
                "rank": i + 1,
                "title": a["title"],
                "description": a["description"],
                "pubDate": a["pubDate"],
                "link": a["link"],
                "score": a["distance"],
            }
            for i, a in enumerate(pre_rerank)
        ],
        "post_rerank_results": [
            {
                "rank": i + 1,
                "title": a["title"],
                "description": a["description"],
                "pubDate": a["pubDate"],
                "link": a["link"],
                "score": a["distance"],
            }
            for i, a in enumerate(post_rerank)
        ],
        "prompt": prompt,
        "answer": answer,
    }

    trace_dir = "traces"
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"trace_{timestamp_str}.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2, ensure_ascii=False)

    console.print(Panel(answer, title="Answer", border_style="green"))
    console.print("\n[bold]Pre-rerank candidates:[/bold]")
    for r in pre_rerank:
        console.print(f"  [{r['distance']:.4f}] {r['title']}")
    console.print("\n[bold]Post-rerank results:[/bold]")
    for r in post_rerank:
        console.print(f"  [{r['distance']:.4f}] {r['title']}")
    console.print(f"\n[bold]Trace saved:[/bold] [cyan]{trace_path}[/cyan]")


@app.command()
def evaluate(
    questions_file: str = typer.Argument(..., help="Path to JSON file with evaluation questions"),
    context: int = typer.Option(5, "--context", "-c", help="Number of articles for context per question"),
    ollama_concurrency: int = typer.Option(3, "--ollama-concurrency", help="Max concurrent Ollama requests"),
    eval_concurrency: int = typer.Option(5, "--eval-concurrency", help="Max concurrent RAGAS evaluation requests"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: semantic, keyword, or hybrid"),
    reranker: str = typer.Option("none", "--reranker", help="Reranker model: none, light, or heavy"),
    notes: str = typer.Option("", "--notes", "-n", help="Description of what changed for the changelog"),
    beta: float = typer.Option(None, "--beta", "-b", help="Hybrid search beta (0.0=keyword, 1.0=semantic); default uses config value (0.7)"),
):
    """Evaluate RAG quality using RAGAS (Faithfulness & Response Relevancy).

    Runs the full pipeline in four phases:
    1. Load evaluation questions from a JSON file.
    2. Run the RAG pipeline (retrieve + generate) for each question.
    3. Score each result with RAGAS Faithfulness and Response Relevancy
       metrics using GPT-4.1 as the evaluator LLM.
    4. Display per-question scores and a summary table, and save results
       to timestamped JSON/CSV logs and the CHANGELOG.md.

    Args:
        questions_file: Path to a JSON array of question strings.
        context: Number of articles to retrieve per question.
        ollama_concurrency: Max concurrent Ollama generation requests.
        eval_concurrency: Max concurrent RAGAS evaluation requests.
        mode: Search mode — 'semantic', 'keyword', or 'hybrid'.
        reranker: Reranker model — 'none', 'light', or 'heavy'.
        notes: Free-text description appended to the changelog entry.
    """
    from src.evaluation import load_questions, async_run_rag_pipeline, async_evaluate_results, save_evaluation_log, append_changelog

    beta_str = f" | Beta: {beta}" if mode == "hybrid" and beta is not None else ""
    console.print(f"[dim]Mode: {mode}{beta_str} | Reranker: {reranker}[/dim]\n")

    # Phase 1: Load questions
    try:
        questions = load_questions(questions_file)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Error loading questions: {e}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold]Loaded {len(questions)} evaluation questions from {questions_file}[/bold]\n")

    # Phase 2: Run RAG pipeline
    def progress_callback(i, total, question):
        console.print(f"  [dim]({i + 1}/{total})[/dim] {question}")

    console.print("[bold]Running RAG pipeline...[/bold]")
    results = asyncio.run(async_run_rag_pipeline(
        questions,
        context_limit=context,
        ollama_concurrency=ollama_concurrency,
        progress_callback=progress_callback,
        search_mode=mode,
        reranker_model=reranker,
        beta=beta,
    ))
    console.print("[green]RAG pipeline complete.[/green]\n")

    # Phase 3: Run RAGAS evaluation
    with console.status("[bold green]Running RAGAS evaluation (this may take a while)..."):
        report = asyncio.run(async_evaluate_results(results, max_concurrency=eval_concurrency))

    console.print("[green]Evaluation complete.[/green]\n")

    # Save evaluation log
    json_path, csv_path = save_evaluation_log(report)
    console.print(f"[bold]Evaluation log saved:[/bold]")
    console.print(f"  JSON: [cyan]{json_path}[/cyan]")
    console.print(f"  CSV:  [cyan]{csv_path}[/cyan]")

    changelog_path = append_changelog(report, search_mode=mode, notes=notes)
    console.print(f"  Log:  [cyan]{changelog_path}[/cyan]\n")

    # Phase 4: Display results
    for i, result in enumerate(report.results, 1):
        truncated_answer = result.answer[:300] + ("..." if len(result.answer) > 300 else "")
        faith_str = f"{result.faithfulness:.4f}" if not math.isnan(result.faithfulness) else "N/A"
        rr_str = f"{result.response_relevancy:.4f}" if not math.isnan(result.response_relevancy) else "N/A"

        panel = Panel(
            f"[bold]Q:[/bold] {result.question}\n\n"
            f"[bold]A:[/bold] {truncated_answer}\n\n"
            f"[cyan]Faithfulness:[/cyan] {faith_str}    "
            f"[cyan]Response Relevancy:[/cyan] {rr_str}",
            title=f"Question {i}",
            border_style="blue",
        )
        console.print(panel)

    # Summary table
    def rating(score):
        if math.isnan(score):
            return "N/A"
        if score >= 0.8:
            return "[green]Good[/green]"
        if score >= 0.5:
            return "[yellow]Fair[/yellow]"
        return "[red]Poor[/red]"

    def score_str(score):
        return f"{score:.4f}" if not math.isnan(score) else "N/A"

    summary = Table(title="Evaluation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Average Score")
    summary.add_column("Rating")

    summary.add_row(
        "Faithfulness",
        score_str(report.avg_faithfulness),
        rating(report.avg_faithfulness),
    )
    summary.add_row(
        "Response Relevancy",
        score_str(report.avg_response_relevancy),
        rating(report.avg_response_relevancy),
    )

    console.print("\n")
    console.print(summary)


if __name__ == "__main__":
    app()
