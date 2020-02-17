# PyTorch Dataset class for Hierarchical Attention Models

from torch.utils.data import Dataset
from vectorizer import Vectorizer
import os
import json
import numpy as np
from typing import List

class HAND(Dataset):

    def __init__(self, snippets: List[str], targets: list, top_n: int, device="cpu", *kwargs):
        """
        Instantiate a Dataset object for documents to be used with the HAN ("HAN Data" --> "HAND")
        :param snippets: List of input documents. Documents are allowed to be of different lengths.
        :param targets: labels associated with each snippet
        :param vectorizer: the vectorizer used to process the data
        :param top_n: number of words/tokens to retain (e.g. top 10.000 words by frequency)
        :return: object of class HAND
        """
        # Make lookup dictionary
        self._vectorizer, snippets = Vectorizer.from_dict(snippets, targets, top_n)
        # Vectorize
        self._snippets_vectorized = [self._vectorizer.to_sequences(doc, device=device) for doc in snippets]
        # Save targets
        if isinstance(targets[0], int):
            self._targets = targets
            self._vocab_targets = False
        else:
            self._targets = [self._vectorizer.map_labels(target, device=device) for target in targets]
            self._vocab_targets = True
        # Length
        self._len = len(snippets)

    @staticmethod
    def load_vectorizer(path: str):
        # Assert file exists
        assert os.path.exists(path), "Path '{}' does not exist".format(path)
        # Load
        Vectorizer.from_serializable(path)

    @staticmethod
    def load_data(path: str):
        # Assert file exists
        assert os.path.exists(path), "Path '{}' does not exist".format(path)

    # TODO:
    #  - to serializable
    #  - from seralizable
    #  - ...
    #  - label mapping

    def __getitem__(self, index: int):
        """
        the primary entry point method for PyTorch datasets. PyTorch needs this to be able to access data for training
        :param index (int): the index to the data point
        :return: a dictionary holding the data point's features (x_data) and label (y_target)
        """ 
        return()


    def __len__(self):
        """Standard method to get length (i.e. number of snippets)"""
        return self._len

    def len_batches(self, batch_size: int)->"int":
        """
        Given a batch size, return the number of batches in the dataset
        :param batch_size: size of each batch
        :return: number of batches in the dataset
        """
        return len(self) // batch_size
