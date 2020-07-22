import numpy as np
import json
import spacy
from collections import Counter
nlp = spacy.load("en_core_web_sm")

# Input words output np arrays of vocabulary positions
class Vectorizer(object):

    def __init__(self, vocabulary_text, vocabulary_labels, top_n = 10000):
        """Set up the vectorizer"""
        self._vocabulary_text = vocabulary_text
        self._vocabulary_labels = vocabulary_labels
        # Subtract 1 for <UNK> token
        self._top_n = top_n-1
        # Get the most common words as list
        self._most_common = vocabulary_text.most_common(self._top_n)
        # Filter vocabulary to only keep the top_n words
        if len(self._vocabulary_text._token_to_idx) > top_n:
            # Retrieve most common terms
            mc = [*self._most_common.keys()]
            # Add unknown token if not in most common
            if self._vocabulary_text._unk_token not in mc:
                mc.insert(0, self._vocabulary_text._unk_token)
            # Keep only the most common words
            self._vocabulary_text._token_to_idx = {k:i for i, k in enumerate(mc)}
            self._vocabulary_text._idx_to_token = {i:k for i, k in self._vocabulary_text._token_to_idx.items()}
            self._TF = {k:self._vocabulary_text._TF[k] for k in mc}

    def to_sequence(self, text: str):
        """
        Turn a piece of text into a sequence of indices
        :param text: text to turn into a sequence
        """
        # Assert len of text > 0
        if len(text) == 0:
            print(text)
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

    def to_sequences(self, texts: list):
        """
        Take many texts, turn into sequences and make into matrix
        :param texts: list of lists of texts to vectorize
        """
        # For each document, vectorize the list entries (sentences)
        texts_as_sequences = [self.to_sequence(sentence) for sentence in texts if len(sentence) > 0]
        # Return
        return(texts_as_sequences)

    def map_label(self, label: str):
        """
        Map a single label
        :param label: label to vectorize
        """
        return self._vocabulary_labels.lookup_token(label)

    def map_labels(self, labels: list):
        """
        Take labels and turn vectorize
        :param labels: outcome labels, as list
        """
        labels_vectorized = [self.map_label(label) for label in labels]
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
        # Add labels to vocabulary
        for l in set(targets):
            vocabulary_labels.add_token(l)
        # Do nlp on texts
        with nlp.disable_pipes("ner"):
            for snippet in nlp.pipe(snippets):
                # To hold sentences
                doc = []
                # For each sentence, get tokens unless it is punctuation or a number
                for sentence in snippet.sents:
                    current_sentence = " ".join([token.text for token in sentence if token.pos_ not in ["NUM", "PUNCT"]])
                    doc.append(current_sentence)
                    # Add to vocab
                    vocabulary_text.add_many(current_sentence.split())
                # Add to segmented texts
                txt_segmented.append(doc)
        # Return class
        return(cls(vocabulary_text, vocabulary_labels, top_n), txt_segmented)

# Create a vocabulary from input texts
class Vocabulary(object):

    """Create word to integer mapping and vice versa"""

    def __init__(self, token_to_idx = None, TF = None, use_unknown_token = True, unk_token = "<UNK>"):
        # If does not exist, then create
        if token_to_idx is None:
            token_to_idx = {}
        # Keep track of Term Frequencies
        if TF is None:
            self._TF = Counter()
        else:
            self._TF = Counter(TF)
        # Save index
        self._token_to_idx = token_to_idx
        # Reverse index
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}
        # Unknown word token
        self._unk_token = unk_token
        self._use_unknown_token = use_unknown_token
        self.unk_index = 0
        if use_unknown_token:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """ returns a dictionary that can be serialized """
        return({'token_to_idx': self._token_to_idx,
                'TF': dict(self._TF),
                'use_unknown_token': self._use_unknown_token,
                'unk_token': self._unk_token})

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """
        #(** unzips the file)
        return(cls(**contents))

    def add_token(self, token: str):
        """Add a token to the vocabulary"""
        if self._token_to_idx.get(token) is not None:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        # Add to TF
        self._TF[token] += 1
        return(index)

    def add_many(self, tokens: list):
        """Add many tokens to the vocabulary"""
        return([self.add_token(token) for token in tokens])

    def lookup_token(self, token: str):
        """Look up a token in the index"""
        return(self._token_to_idx.get(token, self.unk_index))

    def lookup_index(self, index: int):
        """Use an index value to get a token"""
        if self._idx_to_token.get(index) is None:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return(self._idx_to_token[index])

    def TF(self, token: str):
        """Get the term frequency of a token"""
        if self._TF.get(token) is None:
            return(0)
        else:
            return(self._TF[token])

    def most_common(self, top_n: int):
        """Get the most common terms in the vocabulary"""
        return({inp[0]:inp[1] for inp in self._TF.most_common(top_n)})

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)
