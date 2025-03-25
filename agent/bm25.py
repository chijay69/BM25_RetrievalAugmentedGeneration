import asyncio
import math
import math
import os.path
import pickle
from itertools import groupby
from typing import List, Tuple

from smolagents import Tool

from bm25Tool.build_document_index import read_file_content
from bm25Tool.load_build_retriever_file import load_or_build_retriever_state
from bm25Tool.query_bm25 import query_bm25_tool
from config_reader import get_base_directory, get_output_dir, get_retriever_file, get_bm25_parameters
from converter import Document
from converter.clean_text import clean_text

BASE_DIR: str = get_base_directory()
CONFIG_PATH: str = os.path.join(BASE_DIR, "config.ini")
output_path: str = os.path.join(BASE_DIR, get_output_dir(CONFIG_PATH))
retriever_file: str = os.path.join(BASE_DIR, get_retriever_file(CONFIG_PATH))

b, k1 = get_bm25_parameters(CONFIG_PATH)


class BM25Tool(Tool):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_path
        self.retriever_file = retriever_file
        self.documents = []
        self.avgdl = 0
        self.N = 0
        self.term_document_freq = {}
        self.k1 = k1
        self.b = b
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

    @staticmethod
    def bm25_score(query: str)-> List[Tuple[Document, float]]:
        """
        Calculates the bm25 score for a document in relevance to the query.
        :param query: User input (question or request).
        :return: returns a list of Tuple containing the document and its score
        """
        doc_scores: List = query_bm25_tool(query)
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)

    def forward(self, query: str, num_snippets: int = 5):
        return self.main(query, num_snippets)

    def main(self, query: str, num_snippets: int = 5):
        num_snippets = min(num_snippets, 5)
        if not query:
            return ""
        results = self.bm25_score(query)
        results = results[:num_snippets]
        results.sort(key=lambda doc_: (doc_[0].metadata["filename"]))
        grouped_results = groupby(results, key=lambda  doc_: doc_[0].metadata["filename"])

        output = []

        for doc_name, group in grouped_results:
            output.append(f"============================{doc_name}============================")
            toc_filename: str = doc_name.rsplit(".", 1)[0] + "_toc.md"

            toc_file_path = os.path.join(output_path, toc_filename)

            if os.path.exists(toc_file_path):
                toc_content: str = read_file_content(toc_file_path)
                output.append("Table of Content\n")
                output.append(toc_content)
                output.append("\n===========\n")

            for doc, score in group:
                section_title = doc.metadata.get("section", "Unknown Section")
                output.append(f"Section: {section_title}")
                snippet_content = "\n".join(doc.chunk_content.split("\n")[3:])
                output.append(f"\nRelevant Snippet: \n{snippet_content}\nScore: {score:.1f}")
                output.append("\n==============\n")
            output.append("\n========================================\n")
        return "\n".join(output)