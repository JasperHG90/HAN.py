# PyTorch Dataset class for Wikipedia articles processed by WikiEssentials
# Will remove preprocessing steps from WikiEssentials and to all the preprocessing here
# Remember to accept punctuation in syntok segmenter

from torch.utils.data import Dataset
from utils.vectorizer import Vectorizer
import os
import json

from pydantic import BaseModel
from typing import List

class WikiDataset(Dataset):

    def __init__(self, snippets: dict, targets: list, vectorizer: Vectorizer, top_n: int):

        """
        Instantiate a Dataset object for wikipedia data
        :param snippets: dictionary with Wikipedia data, processed at snippet ("paragraph") level.
        :param targets: labels associated with each snippet
        :param vectorizer: the vectorizer used to process the data
        :param top_n: number of words/tokens to retain (e.g. top 10.000 words by frequency)
        :return: object of class WikiDataset
        """

        # Save inputs
        self.snippets = snippets
        self._vectorizer = vectorizer

        # Make lookup dictionary
        self._vectorizer.from_dict(snippets, targets, top_n)

        # Save targets
        self._targets = targets

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

    @staticmethod
    def clean_text():

        """
        Clean bunch of input texts for standard stuff
        :param remove_digits:
        :param to_lower:
        :param remove_special:
        :param fix_contractions:
        :param fix_spelling:
        """

        pass

    def __getitem__(self, index: int):
        """
        the primary entry point method for PyTorch datasets. PyTorch needs this to be able to access data for training
        :param index (int): the index to the data point
        :return: a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        review_vector = \
            self._vectorizer.vectorize(row.review)

        rating_index = \
            self._vectorizer.rating_vocab.lookup_token(row.rating)

        return {'x_data': review_vector,
                'y_target': rating_index}

    def __len__(self):

        """Standard method to get length (i.e. number of snippets)"""

        #return self._target_size
        pass

    def __print__(self):

        """Print method """

        pass

    def len_batches(self, batch_size: int)->"int":
        """
        Given a batch size, return the number of batches in the dataset
        :param batch_size: size of each batch
        :return: number of batches in the dataset
        """
        return len(self) // batch_size
