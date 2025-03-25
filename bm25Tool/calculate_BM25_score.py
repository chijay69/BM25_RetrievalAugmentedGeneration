"""calculate_BM25_score"""
import math
import os.path
from typing import Dict, List

from config_reader import get_bm25_parameters, get_base_directory
from converter.Document import Document
from converter.clean_text import clean_text

BASE_DIR:str = get_base_directory()
CONFIG_PATH:str = os.path.join(BASE_DIR, "config.ini")

k1, b = get_bm25_parameters(CONFIG_PATH) # BM25 parameters

def calculate_bm25_score(query: str, document: Document, N: int, avgdl: float, term_frequency: Dict[str, int]) -> float:
    """
    Calculates the BM25 score for a query and document.
    :param query:
    :param document:
    :param N:
    :param avgdl:
    :param term_frequency:
    :return:
    """
    query_terms: List[str] = clean_text(query).split()
    score: float = 0.0

    for term in query_terms:
        if term in document.term_freq:
            df: int = term_frequency.get(term, 1)
            idf: float = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf: int = document.term_freq[term]
            score: float = score + idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (document.doc_len / avgdl))))

    return score
