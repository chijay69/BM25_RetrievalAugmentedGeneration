import asyncio
import logging
import os
import sys
from logging import Logger

from bm25Tool.load_build_retriever_file import load_or_build_retriever_state
from bm25Tool.print_result import print_results
from bm25Tool.rank_document import rank_documents
from config_reader import get_retriever_file, get_output_dir, get_base_directory

BASE_DIR: str = get_base_directory()
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.ini")
RETRIEVER_FILE: str = os.path.join(BASE_DIR, get_retriever_file(CONFIG_PATH))
OUTPUT_DIRECTORY: str = os.path.join(BASE_DIR, get_output_dir(CONFIG_PATH))
TOP_K: int = 100


async def main() -> None:
    """
    Executes a BM25 query, retrieving and ranking documents based on the input query.
    """

    log_filename: str = os.path.basename(__file__)
    log_path: str = os.path.join(BASE_DIR, "logs", log_filename)

    logging.basicConfig(
        filename=log_path,
        filemode="w+",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger: Logger = logging.getLogger(__name__)

    logger.info("Loading or building retriever state.")
    documents, N, avgdl, term_frequency = await load_or_build_retriever_state(RETRIEVER_FILE)

    query: str = sys.argv[1] if len(sys.argv) > 1 else "Summarize the text"
    show_full_text: bool = "--test" not in sys.argv

    logger.info("Ranking documents based on the query.")
    results = rank_documents(query, documents, N, avgdl, term_frequency)[:TOP_K]

    logger.info("Printing the query results.")
    await print_results(results, OUTPUT_DIRECTORY, show_full_text)


if __name__ == "__main__":
    asyncio.run(main())
