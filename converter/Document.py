from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

from typing_extensions import LiteralString

from converter.clean_text import clean_text


@dataclass
class Document:
    """
    A class representation of a document object.
    """
    chunk_content: str
    metadata: Dict[str, str]
    clean_terms: List[str] = field(default_factory=list, init=False)
    term_freq: Dict[str, int] = field(default_factory=dict, init=False)
    doc_len: int = field(default=0, init=False)

    def __post_init__(self):
        """
        Initializes base attributes.
        """
        self.update_derived_attributes()


    def __str__(self):
        """
        Returns a string representation of the Document object.
        """
        return f"Document(content='{self.chunk_content[:50]}...', metadata={self.metadata})"

    def to_dict(self):
        """
        Returns a dictionary representation of the Document object.
        """
        return {
            "chunk_content": self.chunk_content,
            "metadata": self.metadata,
        }

    def compute_clean_terms(self) -> List[str]:
        """
        Computes and returns the clean terms from a text.
        """
        self.clean_terms: List[str] = clean_text(self.chunk_content).split()
        return self.clean_terms

    def compute_term_freq(self) -> Dict[str, int]:
        """
        Computes and returns the term frequency.
        """
        self.term_freq = Counter(self.clean_terms)
        return self.term_freq

    def compute_doc_len(self) -> int:
        """
        Computes and returns the document length.
        """
        self.doc_len = len(self.clean_terms)
        return self.doc_len

    def update_derived_attributes(self):
      """
      Computes all derived attributes.
      """
      self.compute_clean_terms()
      self.compute_term_freq()
      self.compute_doc_len()

