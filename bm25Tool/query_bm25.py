import os
import sys
from logging import Logger

from smolagents import tool

from bm25Tool.create_and_save_toc import create_toc
from bm25Tool.load_build_retriever_file import load_or_build_retriever_state
from bm25Tool.print_result import print_results
from bm25Tool.rank_document import rank_documents
from bm25Tool.setup_logger import setup_logger
from config_reader import get_retriever_file, get_output_dir, get_base_directory
from converter.Document import Document

BASE_DIR: str = get_base_directory()
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.ini")
RETRIEVER_FILE: str = os.path.join(BASE_DIR, get_retriever_file(CONFIG_PATH))
OUTPUT_DIRECTORY: str = os.path.join(BASE_DIR, get_output_dir(CONFIG_PATH))
TOP_K: int = 100


def query_bm25_tool(query: str = None) -> list[tuple[Document, float]]:
    """
    Executes a BM25 query, retrieving and ranking documents based on the input query.
    """

    logger: Logger = setup_logger(__name__)

    logger.info("Loading or building retriever state.")
    documents, N, avgdl, term_frequency = load_or_build_retriever_state(RETRIEVER_FILE)

    query: str = query if query else "Summarize the text"
    show_full_text: bool = "--test" not in sys.argv

    logger.info("Generate toc file")
    create_toc(OUTPUT_DIRECTORY)

    logger.info("Ranking documents based on the query.")
    results = rank_documents(query, documents, N, avgdl, term_frequency)[:TOP_K]

    logger.info("Printing the query results.")
    print_results(results, OUTPUT_DIRECTORY, show_full_text)

    return results

if __name__=="__main__":
    query_bm25_tool()
