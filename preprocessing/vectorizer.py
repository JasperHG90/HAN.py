# Input words output np arrays of vocabulary positions

import numpy as np
from vocabulary import Vocabulary
import json
import syntok.segmenter as segmenter
import torch

class Vectorizer(object):

    def __init__(self, vocabulary_text, vocabulary_labels, top_n = 10000):
        """Set up the vectorizer"""
        self._vocabulary_text = vocabulary_text
        self._vocabulary_labels = vocabulary_labels
        self._top_n = top_n
        # Get the most common words as list
        self._most_common = vocabulary_text.most_common(self._top_n)

    def to_sequence(self, text: str):
        """
        Turn a piece of text into a sequence of indices
        :param text: text to turn into a sequence
        """
        # Assert len of text > 0
        assert len(text) > 0, "Text too small to process"
        # List 
        text_sequence = [0] * len(text.split())
        # For each token, retrieve the index of the token and store
        for i, token in enumerate(text.split()):
            # If  in most common
            # (else value is already 0)
            if self._most_common.get(token) is not None:
                text_sequence[i] = self._vocabulary_text.lookup_token(token)
        # Return
        return(text_sequence)

    def to_sequences(self, texts: list, device = "cpu"):
        """
        Take many texts, turn into sequences and make into matrix
        :param texts: list of lists of texts to vectorize
        :param device: either one of "cpu" or "cuda"
        """
        # For each document, vectorize the list entries (sentences)
        texts_as_sequences = [torch.tensor(self.to_sequence(sentence)).type(torch.long).to(device) for sentence in texts]
        # Return
        return(texts_as_sequences)

    def map_label(self, label):
        """
        Map a single label
        :param label: label to vectorize
        """
        return self._vocabulary_labels.lookup_token(label)

    def map_labels(self, labels: list, device = "cpu"):
        """
        Take labels and turn vectorize
        :param labels: outcome labels, as list
        :param device: either one of "cpu" or "cuda"
        """
        labels_vectorized = [torch.tensor(self.map_label(label)).type(torch.long).to(device) for label in labels]
        return(labels_vectorized)
        
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
        
        :return: tuple containing the vectorizer class and segmented snippets
        """
        # Initialize vocabularies for labels
        vocabulary_text = Vocabulary(use_unknown_token = True)
        vocabulary_labels = Vocabulary(use_unknown_token = False)
        # To hold segmented texts
        txt_segmented = []
        # Populate vocabulary
        for label, snippet in zip(targets, snippets):
            # To hold sentences
            doc = []
            # Segment input
            a = segmenter.process(snippet)
            for par in a:
                for sent in par:
                    # HERE! 
                    csent = "".join([token.spacing + token.value for token in sent]).strip()
                    # Add to sentences
                    doc.append(csent)
                    # Add to vocab
                    vocabulary_text.add_many(csent.split())
            # Add to segmented texts
            txt_segmented.append(doc)
            # Add labels
            vocabulary_labels.add_token(label)
        # Return class
        return(cls(vocabulary_text, vocabulary_labels, top_n), txt_segmented)
