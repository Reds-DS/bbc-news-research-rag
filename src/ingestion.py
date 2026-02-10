import pandas as pd
from langchain_core.documents import Document
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from src.config import BATCH_SIZE, DATA_PATH
from src.database import ChromaDB


def load_csv(path: str = DATA_PATH) -> pd.DataFrame:
    """Load BBC News CSV file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def prepare_documents(df: pd.DataFrame) -> list[Document]:
    """Convert DataFrame rows into LangChain Documents with metadata."""
    documents = []

    for _, row in df.iterrows():
        title = "" if pd.isna(row.title) else str(row.title)
        desc = "" if pd.isna(row.description) else str(row.description)
        page_content = f"{title}\n\n{desc}".strip()
        metadata = {
            "title": title,
            "description": desc,
            "pubDate": "" if pd.isna(row.pubDate) else str(row.pubDate),
            "guid": "" if pd.isna(row.guid) else str(row.guid),
            "link": "" if pd.isna(row.link) else str(row.link),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def ingest(csv_path: str = DATA_PATH, recreate: bool = False):
    """Load articles from CSV, embed them, and store in ChromaDB."""
    db = ChromaDB()

    if recreate and db.collection_exists():
        db.delete_collection()
        print(f"Deleted existing collection")

    df = load_csv(csv_path)
    print(f"Loaded {len(df)} articles from {csv_path}")

    documents = prepare_documents(df)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Embedding and inserting into Chroma...", total=len(documents))

        for i in range(0, len(documents), BATCH_SIZE):
            batch = documents[i:i + BATCH_SIZE]
            db.add_documents(batch)
            progress.update(task, advance=len(batch))

    count = db.get_count()
    print(f"Ingestion complete. Total objects: {count}")
