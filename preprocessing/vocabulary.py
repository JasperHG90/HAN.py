# Create a vocabulary from input texts
from collections import Counter

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
