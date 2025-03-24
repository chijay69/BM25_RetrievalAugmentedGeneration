# load_build_retriever_file.py
import logging
import os.path
import pickle
from typing import Any, Dict

from bm25Tool.build_document_index import build_document_index
from config_reader import get_base_directory, get_data_dir, get_output_dir

BASE_DIR = get_base_directory()

CONFIG_PATH = os.path.join(BASE_DIR, "config.ini")
DATA_PATH = os.path.join(BASE_DIR, get_data_dir(CONFIG_PATH))
OUTPUT_PATH = os.path.join(BASE_DIR, get_output_dir(CONFIG_PATH))
LOG_PATH = os.path.join(BASE_DIR, 'logs/build_document.log')


async def load_or_build_retriever_state(retriever_file: str = None, refresh: bool = False)-> tuple[Any, Any, Any, Any]:
    """
    Loads a retriever file from the path provided.
    :param refresh: boolean
    :param retriever_file: The retriever file path.
    :return: the retriever file content as string.
    """
    try:
        if retriever_file is None:
            raise ValueError("Retriever file cannot be None")
        if os.path.exists(retriever_file) and refresh:
            with open(retriever_file, "rb", encoding="utf-8") as file:
                state: pickle = pickle.load(file)
                return state.get("documents", []), state.get("N", 0), state.get("avgdl", 0), state.get("term_document_freq", {})

        documents, term_frequency = await build_document_index(input_dir=DATA_PATH, output_dir=OUTPUT_PATH)
        N: int = len(documents) if documents else 0
        avgdl: float = sum(doc.doc_len for doc in documents)

        state: Dict = {"documents": documents, "avgdl": avgdl, "N": N, "term_document_freq": term_frequency}

        with open(retriever_file, "w+b") as f:
            pickle.dump(state, f)

        return documents, N, avgdl, term_frequency
    except ValueError as e:
        logging.error(e)
        raise
    except Exception as e:
        logging.error(e)
        raise

