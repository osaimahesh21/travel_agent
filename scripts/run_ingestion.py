from travel_assistant.core.config import Settings
from travel_assistant.services.ingestion_service import IngestionService
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def main():
    settings = Settings()
    ingestion = IngestionService(settings)


    pdfs = ingestion.list_pdfs("/Users/saimaheshobillaneni/Desktop/travel_agent_project/src/travel_assistant/data/docs")
    print("PDFs found:", pdfs)

    pages = ingestion.load_pdf_pages(pdfs[0])
    print("Number of pages:", len(pages))

    first_page = pages[0]
    print("\n--- First page metadata ---")
    print(first_page.metadata)

    print("\n--- First page text preview ---")
    print(first_page.page_content[:300])

    result = ingestion.ingest_directory_to_jsonl(
    docs_dir="/Users/saimaheshobillaneni/Desktop/travel_agent_project/src/travel_assistant/data/docs",
    output_path="ingestion_output/chunks.jsonl"   # <-- your new output path goes here
)

if __name__ == "__main__":
    main()