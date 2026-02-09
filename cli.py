import asyncio
import json
import math

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="BBC News RAG System")
console = Console()


@app.command()
def status():
    """Check Chroma and Ollama connection status."""
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
    """Load BBC News data into Chroma."""
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
):
    """Search for news articles by semantic similarity."""
    from src.retrieval import Retriever

    console.print(f"[dim]Mode: {mode}[/dim]\n")
    console.print(f"[bold]Searching for:[/bold] {query}\n")

    retriever = Retriever(mode=mode)
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
):
    """Ask a question using RAG (retrieval + generation)."""
    from src.retrieval import Retriever
    from src.generation import RAGGenerator

    console.print(f"[dim]Mode: {mode}[/dim]\n")
    console.print(f"[bold]Question:[/bold] {question}\n")

    with console.status("[bold green]Searching for relevant articles..."):
        retriever = Retriever(mode=mode)
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
def evaluate(
    questions_file: str = typer.Argument(..., help="Path to JSON file with evaluation questions"),
    context: int = typer.Option(5, "--context", "-c", help="Number of articles for context per question"),
    ollama_concurrency: int = typer.Option(3, "--ollama-concurrency", help="Max concurrent Ollama requests"),
    eval_concurrency: int = typer.Option(5, "--eval-concurrency", help="Max concurrent RAGAS evaluation requests"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: semantic, keyword, or hybrid"),
):
    """Evaluate RAG quality using RAGAS (Faithfulness & Response Relevancy)."""
    from src.evaluation import load_questions, async_run_rag_pipeline, async_evaluate_results, save_evaluation_log

    console.print(f"[dim]Mode: {mode}[/dim]\n")

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
    console.print(f"  CSV:  [cyan]{csv_path}[/cyan]\n")

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
