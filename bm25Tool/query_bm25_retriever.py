import math
import os
import pathlib
import pickle
import re
import sys
from itertools import groupby
from typing import List, Any, Set, LiteralString, Dict, Tuple

import nltk

from config import OUTPUT_DIRECTORY, RETRIEVER_FILE, CHUNK_SIZE, LANGUAGE, TOP_K
from converter.Document import Document
from converter.clean_text import clean_text


# Get the project root directory (directory containing config.py)
project_root = pathlib.Path(__file__).resolve().parts
project_root1 = "\\".join(part.rstrip("\\") if part.endswith("\\") else part for part in project_root[:-2])


# Construct the output_directory path relative to the project root
# output_directory = project_root.joinpath(OUTPUT_DIRECTORY)
output_directory = os.path.join(project_root1, OUTPUT_DIRECTORY)
print(output_directory)

var = [print(filename) for filename in os.listdir(output_directory)]

retriever_file = RETRIEVER_FILE
chunk_size = CHUNK_SIZE
k1, b = 1.5, 0.75  # BM25 parameters

def read_file_content(filepath: str) -> str:
    """Reads the content of a file."""
    with open(filepath, 'r') as file:
        return file.read()

def split_content_into_sections(content: str) -> List[Set[str | None | Any]]:
    """Splits content into sections based on markdown headers."""
    sections = []
    current_section = []
    section_title = None

    for line in content.splitlines():
        match = re.match(r'^(#+)\$+(.*)', line)
        if match and len(match.group(1)) > 1 and len(match.group(2).strip()) >= 5:
            if current_section:
                sections.append({section_title, '\n'.join(current_section[1:])})
                current_section = []
            section_title = match.group(2).strip()
        current_section.append(line)

    if current_section:
        sections.append({section_title, '\n'.join(current_section[1:])})
    return sections

def tokenize_sentences(text: str) -> List[str]:
    """Tokenizes text into sentences."""
    return nltk.sent_tokenize(text, LANGUAGE)

def create_document_chunk(metadata: Dict[str, str], chunk_content: str) -> Document:
    """Creates a Document object from chunk content and metadata."""
    return Document(chunk_content=chunk_content, metadata=metadata)

def split_section_into_chunks(section: Set[LiteralString], metadata: Dict[str, str]) -> List[Document]:
    """Splits a section into chunks of sentences."""
    section_title, section_content = section
    sentences = tokenize_sentences(section_content)
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size:
            chunk_content = f"Document: {metadata['filename']}\nSection: {section_title}\nSnippet: {' '.join(current_chunk)}"
            chunks.append(create_document_chunk(metadata, chunk_content))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunk_content = f"Document: {metadata['filename']}\nSection: {section_title}\nSnippet: {' '.join(current_chunk)}"
        chunks.append(create_document_chunk(metadata, chunk_content))

    return chunks

def build_document_index(output_directory1: str) -> Tuple[List[Document], Dict[str, int]]:
    """Builds an index of documents and term frequencies."""
    documents = []
    term_document_freq = {}

    for filename in os.listdir(output_directory1):
        if filename.endswith(".md") and not filename.endswith('_toc.md'):
            filepath = os.path.join(output_directory1, filename)
            content = read_file_content(filepath)
            sections = split_content_into_sections(content)

            for section_title, section_content in sections:
                metadata = {"filename": filename, "section": section_title}
                chunks = split_section_into_chunks({section_title, section_content}, metadata)

                for chunk in chunks:
                    chunk.update_derived_attributes()
                    unique_terms = set(chunk.clean_terms)
                    for term in unique_terms:
                        term_document_freq[term] = term_document_freq.get(term, 0) + 1
                documents.extend(chunks)

    return documents, term_document_freq

def calculate_bm25_score(query: str, document: Document, N: int, avgdl: float, term_document_freq: Dict[str, int]) -> float:
    """Calculates the BM25 score for a query and document."""
    query_terms = clean_text(query).split()
    score = 0.0

    for term in query_terms:
        if term in document.term_freq:
            df = term_document_freq.get(term, 1)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf = document.term_freq[term]
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (document.doc_len / avgdl))))

    return score

def rank_documents(query: str, documents: List[Document], N: int, avgdl: float, term_document_freq: Dict[str, int]) -> List[Tuple[Document, float]]:
    """Ranks documents based on BM25 score."""
    doc_scores = []
    for doc in documents:
        score = calculate_bm25_score(query, doc, N, avgdl, term_document_freq)
        doc_scores.append((doc, score))
    return sorted(doc_scores, key=lambda x: x[1], reverse=True)

def load_or_build_retriever_state(output_directory2: str, retriever_file: str, refresh: bool = False) -> Tuple[List[Document], int, float, Dict[str, int]]:
    """Loads or builds the retriever state."""
    if os.path.exists(retriever_file) and not refresh:
        with open(retriever_file, 'rb') as f:
            state = pickle.load(f)
            return state.get("documents", []), state.get("N", 0), state.get("avgdl", 0), state.get("term_document_freq", {})

    documents, term_document_freq = build_document_index(output_directory2)
    N = len(documents) if documents else 0
    avgdl = sum(doc.doc_len for doc in documents) / N if documents else 0

    state = {"documents": documents, "avgdl": avgdl, "N": N, "term_document_freq": term_document_freq}
    with open(retriever_file, 'w+b') as f:
        pickle.dump(state, f)

    return documents, N, avgdl, term_document_freq

def print_results(results: List[Tuple[Document, float]], output_directory3: str, show_full_text: bool):
    """Prints the search results."""
    results.sort(key=lambda doc: doc[0].metadata['filename'])
    grouped_result = groupby(results, key=lambda doc: doc[0].metadata['filename'])

    for doc_name, group in grouped_result:
        print(f"========================= {doc_name} ============================")
        toc_file_path = os.path.join(output_directory3, doc_name.rsplit(".", 1)[0] + "_toc.md")
        if os.path.exists(toc_file_path):
            print("Table of Content\n" + read_file_content(toc_file_path) + "\n=====\n")

        for doc, score in group:
            snippet_content = "\n".join(doc.chunk_content.split("\n")[3:])
            snippet = snippet_content if show_full_text else snippet_content[:100] + '...' + snippet_content[-100:]
            print(f"Relevant Snippet:\n{snippet}\nScore: {score:.1f}\n=====\n")

        print("\n============\n")

def main():
    """Main function to execute the search."""
    documents, N, avgdl, term_document_freq = load_or_build_retriever_state(output_directory, retriever_file)

    # query = sys.argv[1] if len(sys.argv) > 1 else ""
    query: str = "Summarize the text"
    if not query:
        print("Please provide a query argument.")
        sys.exit(1)

    show_full_text = '--test' not in sys.argv

    results = rank_documents(query, documents, N, avgdl, term_document_freq)[:TOP_K]
    print_results(results, output_directory, show_full_text)
