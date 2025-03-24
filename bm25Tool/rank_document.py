""" rank_document.py"""
from typing import List, Dict, Tuple

from bm25Tool.calculate_BM25_score import calculate_bm25_score
from converter.Document import Document


def rank_documents(query: str, documents: List[Document], N: int, avgdl: float, term_frequency: Dict[str, int]) -> List[
    Tuple[Document, float]]:
    """
    Ranks documents based on relevance to query.
    :param query:
    :param documents:
    :param N:
    :param avgdl:
    :param term_frequency:
    :return:
    """
    doc_scores: List[Tuple[Document, float]] = []

    for doc in documents:
        score: float = calculate_bm25_score(query, doc, N, avgdl, term_frequency)
        doc_scores.append((doc, score))
    return sorted(doc_scores, key=lambda  x: x[1], reverse=True)


