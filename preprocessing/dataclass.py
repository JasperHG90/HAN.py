# PyTorch Dataset class for Hierarchical Attention Models

from torch.utils.data import Dataset
from vectorizer import Vectorizer
import os
import json
import numpy as np
from typing import List, Union
import torch

class HAND(Dataset):

    def __init__(self, snippets: List[str], targets: List[Union[int, str]], top_n: int, device="cpu", *kwargs):
        """
        Instantiate a Dataset object for documents to be used with the HAN ("HAN Data" --> "HAND")
        :param snippets: List of input documents. Documents are allowed to be of different lengths.
        :param targets: labels associated with each snippet. Can be either a list of integers or string labels.
        :param vectorizer: the vectorizer used to process the data
        :param top_n: number of words/tokens to retain (e.g. top 10.000 words by frequency)
        :return: object of class HAND
        """
        # Make lookup dictionary
        self._device = device
        self._snippets = snippets
        self._vectorizer, snippets = Vectorizer.from_dict(snippets, targets, top_n)
        # Vectorize
        self._snippets_vectorized = [self._vectorizer.to_sequences(doc) for doc in snippets]
        # Save targets
        if isinstance(targets[0], int):
            self._targets = targets
            self._vocab_targets = False
        else:
            self._targets = [self._vectorizer.map_labels(target) for target in targets]
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

    def __getitem__(self, index: int):
        """
        the primary entry point method for PyTorch datasets. PyTorch needs this to be able to access data for training
        :param index (int): the index to the data point
        :return: a dictionary holding the data point's features (x_data) and label (y_target)
        """ 
        return(self._snippets_vectorized[index], self._targets[index])

    def __len__(self):
        """Standard method to get length (i.e. number of snippets)"""
        return self._len

    def split(self, seed: int, prop_validation = 0.1):
        """Split the dataset into train and validation"""
        # Set seed
        np.random.seed(seed)
        # Shuffle
        indices = np.random.permutation(self.__len__())
        # Get proportion
        num_val = int(np.round(self.__len__() * prop_validation, 0))
        train_idx = indices[:self.__len__() - num_val]
        test_idx = indices[(self.__len__() - num_val):]
        # Split
        train_data = [self._snippets[index] for index in train_idx]
        train_label = [self._targets[index] for index in train_idx]
        val_data = [self._snippets[index] for index in test_idx]
        val_label = [self._targets[index] for index in test_idx]
        print(train_data)
        print(train_label)
        # Return HAND
        return (HAND(train_data, train_label, self._vectorizer._top_n, self._device),
                HAND(val_data, val_label, self._vectorizer._top_n, self._device))

    def make_batch(self, batch_size: int):
        """Make a batch of size batch_size"""
        rp = np.random.permutation(self.__len__())[:batch_size]
        # Get X, y
        batch = [self.__getitem__(idx) for idx in list(rp)]
        # Return
        return(batch)
        
