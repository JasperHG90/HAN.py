# Input words output np arrays of vocabulary positions

import numpy as np
from utils.vocabulary import Vocabulary
import json

class Vectorizer(object):

    def __init__(self, vocabulary_text, vocabulary_labels, top_n = 10000):

        """Set up the vectorizer"""

        self._vocabulary_text = vocabulary_text
        self._vocabulary_labels = vocabulary_labels
        self._top_n = top_n

    def to_sequence(self, text: str, max_len: int):

        """
        Turn a piece of text into a sequence of indices
        :param text: text to turn into a sequence
        :param max_len: maximum length of a sequence
        """

        # Assert max_len >= 10
        assert max_len >= 10, "'max_len' must be equal to or larger than 10 tokens"

        # Assert len of text > 0
        assert len(text) > 0, "Text too small to process"

        text_sequence = np.zeros(max_len, dtype = "float32")

        # For each token, retrieve the index of the token and store
        for i, token in enumerate(text.split()):
            text_sequence[i] = self._vocabulary_text.lookup_token(token)

        # Return
        return(text_sequence)

    def to_sequences(self, texts: list, max_len: int):

        """
        Take many texts, turn into sequences and make into matrix
        :param texts:
        :param max_len:
        """

        # Dimensions
        texts_as_sequences = np.array((len(texts), max_len), dtype = "float32")

        # Process each
        for i, text in enumerate(texts):
            texts_as_sequences[i, :] = self.to_sequence(text, max_len)

        # Return
        return(texts_as_sequences)

    def to_serializable(self, path: str):

        """Save a vectorizer to disk"""

        vts = self._vocabulary_text.to_serializable()
        vls = self._vocabulary_labels.to_serializable()

        # Dump
        outFile = {"vocab_text": vts, "vocab_labels": vls, "top_n":self._top_n}

        with open(path, "w") as f:
            json.dump(outFile, f)

    @classmethod
    def from_serializable(cls, path: str):

        """Load a vectorizer from a path"""

        with open(path, "r") as f:
            inFile = json.load(f)

        # Set up vocabularies
        vts = Vocabulary.from_serializable(inFile["vocab_text"])
        vls = Vocabulary.from_serializable(inFile["vocab_labels"])

        # Make vocabularies
        return(cls(vts, vls, inFile["top_n"]))

    @classmethod
    def from_dict(cls, snippets: list, targets: list, top_n: int):

        """
        Instantiate a vectorizer class from a data file
        :param path_snippets: list/dict of text snippets
        :param top_n: select the n most occurring words
        """

        # Initialize vocabularies for labels
        vocabulary_text = Vocabulary(use_unknown_token = True)
        vocabulary_labels = Vocabulary(use_unknown_token = False)

        # Populate vocabulary
        for label, snippet in zip(targets, snippets):
            vocabulary_text.add_many(snippet.split())
            # Add labels
            vocabulary_labels.add_token(label)

        # Return class
        return(cls(vocabulary_text, vocabulary_labels, top_n))
