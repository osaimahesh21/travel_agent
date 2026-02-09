from pathlib import Path
from typing import List
import json
import hashlib
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from travel_assistant.core.config import Settings
from travel_assistant.models.schemas import DocumentChunk


logger = logging.getLogger(__name__)


class IngestionService:
    """Converts raw PDFs into a JSONL chunk dataset for downstream embedding and retrieval.

    Decouples PDF parsing from embedding so each stage can be run, tested, and debugged
    independently. Deterministic chunk IDs enable idempotent re-ingestion.

    Pipeline: ingest_directory_to_jsonl()
        -> list_pdfs() -> load_pdf_pages() -> chunk_pages() -> write_chunks_to_jsonl()

    JSONL row schema:
      {"chunk_id": "<sha256>", "content": "...", "metadata": {source_file, source_path, page_number, chunk_index, ...}}
    """

    def __init__(self, settings: Settings):
        """Changing chunk_size or chunk_overlap will produce different chunk boundaries
        and therefore different chunk_ids — downstream embeddings will need a full re-index."""
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        logger.info(
            "Initialized IngestionService | chunk_size=%s chunk_overlap=%s",
            self.settings.chunk_size,
            self.settings.chunk_overlap,
        )

    def list_pdfs(self, docs_dir: str | Path) -> List[Path]:
        """Return sorted list of PDFs in docs_dir. Sorted for deterministic iteration order."""
        docs_dir = Path(docs_dir)
        logger.info("Listing PDFs in directory: %s", docs_dir)

        if not docs_dir.exists():
            logger.error("Docs directory does not exist: %s", docs_dir)
            raise ValueError(f"Directory {docs_dir} does not exist.")

        pdfs = sorted(docs_dir.glob("*.pdf"))
        logger.info("Found %d PDF(s)", len(pdfs))
        return pdfs

    def load_pdf_pages(self, pdf_path: Path):
        """Load a single PDF into LangChain Document objects (one per page).

        Returns List[Document] where each doc has .page_content (str) and .metadata (dict).
        Note: PDF extraction quality varies by library version and PDF structure —
        changes here cascade into different chunk text and chunk IDs.
        """
        logger.info("Loading PDF pages: %s", pdf_path)

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            logger.info("Loaded %d page(s) from %s", len(pages), pdf_path.name)
            return pages
        except Exception:
            logger.exception("Failed while loading PDF: %s", pdf_path)
            raise

    def make_chunk_id(self, pdf_path: Path, page_number: int, chunk_index: int, content: str) -> str:
        """SHA-256 hash of (path + page + index + content) for deterministic, idempotent chunk IDs.

        Caveat: includes source_path, so moving a PDF to a different directory changes its IDs.
        """
        payload = {
            "source_path": str(pdf_path),
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content": content,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def chunk_pages(self, pages, pdf_path: Path) -> List[DocumentChunk]:
        """Split page-level Documents into overlapping chunks with lineage metadata.

        Metadata contract per chunk: source_file, source_path, page_number, chunk_index, chunk_id.
        """
        logger.info("Chunking pages for: %s", pdf_path.name)

        chunks: List[DocumentChunk] = []

        try:
            for page in pages:
                page_num = page.metadata.get("page_number")
                text = page.page_content

                splits = self.splitter.split_text(text)

                logger.debug(
                    "Chunk stats | pdf=%s page=%s splits=%d",
                    pdf_path.name, page_num, len(splits)
                )

                for i, chunk_text in enumerate(splits):
                    # Start from loader metadata, then overlay our lineage fields.
                    meta = dict(page.metadata)
                    meta.update({
                        "source_file": pdf_path.name,
                        "source_path": str(pdf_path),
                        "page_number": page_num,
                        "chunk_index": i,
                    })

                    # -1 fallback: page_number can be None for malformed PDFs.
                    meta["chunk_id"] = self.make_chunk_id(
                        pdf_path=pdf_path,
                        page_number=int(page_num) if page_num is not None else -1,
                        chunk_index=i,
                        content=chunk_text,
                    )

                    chunks.append(DocumentChunk(content=chunk_text, metadata=meta))

            logger.info("Created %d chunk(s) for %s", len(chunks), pdf_path.name)
            return chunks

        except Exception:
            logger.exception("Failed while chunking PDF: %s", pdf_path)
            raise

    def write_chunks_to_jsonl(self, chunks: List[DocumentChunk], output_path: str | Path) -> Path:
        """Persist chunks to JSONL (one JSON object per line). Creates parent dirs if needed.

        Row schema is kept explicit here rather than using model_dump() so downstream
        readers have a stable contract even if the Pydantic model evolves.
        """
        output_path = Path(output_path)
        logger.info("Writing %d chunks to JSONL: %s", len(chunks), output_path)

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                for c in chunks:
                    row = {
                        "chunk_id": c.metadata.get("chunk_id"),
                        "content": c.content,
                        "metadata": c.metadata,
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            logger.info("Wrote JSONL successfully: %s", output_path)
            return output_path

        except Exception:
            logger.exception("Failed while writing JSONL: %s", output_path)
            raise

    def ingest_directory_to_jsonl(self, docs_dir: str | Path, output_path: str | Path) -> Path:
        """End-to-end ingestion: PDFs in -> JSONL out.

        Idempotent — re-running with the same PDFs and chunk settings produces
        identical chunk_ids, so downstream embedding can upsert without duplicates.
        """
        logger.info("Starting ingestion pipeline | docs_dir=%s output_path=%s", docs_dir, output_path)

        try:
            pdfs = self.list_pdfs(docs_dir)

            all_chunks: List[DocumentChunk] = []
            for pdf_path in pdfs:
                pages = self.load_pdf_pages(pdf_path)
                chunks = self.chunk_pages(pages, pdf_path)
                all_chunks.extend(chunks)

            out = self.write_chunks_to_jsonl(all_chunks, output_path)

            logger.info(
                "Ingestion completed | pdfs=%d chunks=%d out=%s",
                len(pdfs), len(all_chunks), out
            )
            return out

        except Exception:
            logger.exception("Ingestion pipeline failed")
            raise
