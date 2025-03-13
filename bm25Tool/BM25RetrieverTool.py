import math
import os.path
import pickle
from itertools import groupby
from typing import List

from smolagents import Tool

from bm25Tool.query_bm25_retriever import load_or_build_retriever_state, k1, b
from config import OUTPUT_DIRECTORY, RETRIEVER_FILE
from converter import Document
from converter.clean_text import clean_text


class BM25RetrieverTool(Tool):
    """
    A BM25 Tool. It provides functions to help the perform specific actions.
    """
    name = "bm25_retriever"
    description = ("A tool that retrieves the most relevant documents and snippets therefrom, along with the sections the snippet are in, using BM25 scoring. "
                   "This implementation use a basic BM25 retrieval method with the following characteristics:\n     -Text preprocessing: Lower-casing and punctuation removal\n"
                   "     -No stemming or advanced text normalization\n      -Uncased, simple text matching\n     -Scores documents based on term frequency and inverse document frequency\n"
                   "     Note: Queries work best with simple, direct language that matches document text.")
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant document sections and snippets"
        },
        "num_snippets": {
            "type": "integer",
            "description": "The number of relevant search snippets to return, maximum is 5",
            "default": 5,
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self, output_dir: str = "docs/output", retriever_file: str = "bm25_retriever.pkl", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = OUTPUT_DIRECTORY
        self.retriever_file = RETRIEVER_FILE
        self.documents = []
        self.avgdl = 0
        self.N = 0
        self.term_document_freq = {}
        self.k1 = 1.5
        self.b = 0.75
        self.is_initialized = False
        self.load_retriever_state()
        self.is_initialized = True

    def load_retriever_state(self):
        """
        Retrieves the current state of the bm25_retriever.
        :return:
        """
        if os.path.exists(self.retriever_file):
            with open(self.retriever_file, "rb") as file:
                state = pickle.load(file)
                self.documents = state.get("documents", [])
                self.avgdl = state.get("avgdl", 0)
                self.N = state.get("N", 0)
                self.term_document_freq = state.get("term_document_freq", {})

    def forward(self, query: str, num_snippets: int = 5):
        num_snippets = min(num_snippets, 5)
        if not query:
            return ""
        results = self.bm25_score(query,self.documents)[:num_snippets]
        results.sort(key=lambda doc: (doc[0].metadata["filename"]))
        grouped_results = groupby(results, key=lambda  doc: doc[0].metadata["filename"])

        output = []

        for doc_name, group in grouped_results:
            output.append(f"============================{doc_name}============================")
            toc_file_path = os.path.join(self.output_dir, doc_name.rsplit(".",1)[0]+"_toc.md")
            if os.path.exists(toc_file_path):
                with open(toc_file_path, "r") as toc_file:
                    toc_content = toc_file.read()
                    output.append("Table of Content")
                    output.append(toc_content)
                    output.append("\n===========\n")
            for doc, score in group:
                section_title = doc.meta_data.get("section", "Unknown Section")
                output.append(f"Section: {section_title}")
                snippet_content = "\n".join(doc.chunk_content.split("\n")[3:])
                output.append(f"\nRelevant Snippet: \n{snippet_content}\nScore: {score:.1f}")
                output.append("\n==============\n")
            output.append("\n=======================\n")
        return "\n".join(output)

    def bm25_score(self, query: str, documents: Document):
        """
        Calculates the bm25 score for a document in relevance to the query.
        :param query:
        :param documents:
        :return:
        """
        documents, N, avgdl, term_document_freq = load_or_build_retriever_state(OUTPUT_DIRECTORY, RETRIEVER_FILE)
        query_terms: List = clean_text(query).split()
        doc_scores: List = []

        for doc in documents:
            score: int = 0
            for term in query_terms:
                if term in doc.term_freq:
                    # Use precomputed document frequency
                    df = self.term_document_freq.get(term, 1)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    tf = doc.term_freq.get(term, 0)
                    score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc.doc_len / avgdl))))
            doc_scores.append((doc, score))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)




