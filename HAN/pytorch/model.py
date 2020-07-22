#%% Implementation of a Hierarchical Attention Network (HAN)
##
## This implementation is based on the following paper
##
##   Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical 
##     attention networks for document classification. In Proceedings of the 2016 conference 
##     of the North American chapter of the association for computational linguistics: human 
##     language technologies (pp. 1480-1489).
##
## Written by: Jasper Ginn <j.h.ginn@uu.nl>
## Course: Pattern Recognition @ Utrecht University

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
# This is a technical thing
# See stackoverflow:
#   - PyTorch: training with GPU gives worse error than training the same thing with CPU
#torch.backends.cudnn.enabled = False

"""
PyTorch modules:
    The PyTorch modules below implement the HAN with attention. The following modules are added.
        1. Attention: implements the attention mechanism described in Yang et al.
        2. word_encoder: applies a bidirectional GRU and attention to input sentences.
        3. sentence_encoder: applies a bidirectional GRU and attention to output of the word encoder
        4. HAN: implementation of the Hierarchical Attention Network. Calls modules 1-3.
    I've taken some inspiration from the following existing implementations:
        - https://github.com/uvipen/Hierarchical-attention-networks-pytorch
        - https://github.com/pandeykartikey/Hierarchical-Attention-Network
"""

#%% Attention module

class Attention(nn.Module):
    def __init__(self, hidden_size):
        """
        Attention mechanism.
        :param hidden_size: size of the hidden states of the bidirectional GRU
        :seealso: 
            - https://pytorch.org/docs/stable/nn.html#gru for the output size of the GRU encoder (bidirectional)
            - Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016, June). Hierarchical attention networks 
              for document classification. In Proceedings of the 2016 conference of the North American chapter of the 
              association for computational linguistics: human language technologies (pp. 1480-1489).
        """
        super(Attention, self).__init__()
        self._hidden_size = hidden_size
        # Linear layer for the tanh activation (eq. 5 in paper)
        #  (times two because bidirectional)
        self._layer1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        # Linear layer for the softmax activation (eq. 6 in paper)
        self._layer2 = nn.Linear(2 * hidden_size, 2 * hidden_size, bias = False)
    def forward(self, hidden_states):
        """
        Forward pass of the attention mechanism
        :param hidden_states: The hidden states of the input sequence at time T
        
        :return: context vector (weighted GRU output) and attention weights
        """
        # (see equation 5)
        u = torch.tanh(self._layer1(hidden_states))
        # (see equation 6)
        alphas = F.softmax(self._layer2(u), dim=1)
        # --> current dimensions: X x Y x Z
        # Sentence vectors
        # (see equation 7)
        # Apply the attention weights (alphas) to each hidden state
        sentence = torch.sum(torch.mul(alphas, hidden_states), dim=1)
        # Return
        return(sentence, alphas)

#%% Word encoder

class word_encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        """
        Word encoder. This part takes a minibatch of input sentences, applies a GRU and attention
         and returns the sequences.
        :param embedding_size: Size of the word embedding
        :param hidden_size: number of hidden units in the word-level GRU
        """
        super(word_encoder, self).__init__()
        self._hidden_size = hidden_size
        # Word-GRU
        self.GRU = nn.GRU(embedding_size, self._hidden_size,
                        bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(self._hidden_size)
    def forward(self, inputs_embedded, hid_state):
        """
        :param inputs_embedded: word embeddings of the mini batch at time t (sentence x seq_length)
        :return: tuple containing:
            (1) weighted GRU annotations (GRU output weighted by the attention vector)
            (2) [final hidden state of GRU (unweighted), attention weights]
        """
        # Bidirectional GRU
        output_gru, last_hidden_state = self.GRU(inputs_embedded)
        # Unpack packed sequence
        output_padded, output_lengths = pad_packed_sequence(output_gru, batch_first=True)
        # Attention
        output_attention, att_weights = self.attention(output_padded)
        # Return
        return(output_attention.unsqueeze(dim=0), [last_hidden_state, att_weights])

#%% Sentence encoder

class sentence_encoder(nn.Module):
    def __init__(self, word_hidden_size, hidden_size):
        """
        Sentence encoder. This part takes as its input a minibatch of documents which have been created by
         the word encoder. It applies a GRU, attention and returns the weighted GRU output.
        :param word_hidden_size: The number of hidden units of the word encoder.
        :param hidden_size: The number of hidden units used for the sentence encoder.
        """
        super(sentence_encoder, self).__init__()
        self._hidden_size = hidden_size
        # Sentence-GRU
        self.GRU = nn.GRU(word_hidden_size, self._hidden_size,
                          bidirectional=True, batch_first=True)       
        # Attention
        self.attention = Attention(hidden_size)
    def forward(self, encoder_output, hid_state):
        """
        :param encoder_output: output of the word encoder.
        :return: weighted annotations created by the sentence GRU
        """
        # Bidirectional GRU
        output_gru, last_hidden_state = self.GRU(encoder_output)
        # Attention
        output_attention, att_weights = self.attention(output_gru)
        # Return
        # (weighted attention vector, hidden states of the sentences)
        return(output_attention.unsqueeze(dim=0), [last_hidden_state, att_weights])

#%% HAN

class HAN(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 embedding_size: int, 
                 hidden_size_words: int, 
                 hidden_size_sent: int, 
                 batch_size: int, 
                 num_classes: int, 
                 device = "cpu",
                 dropout_prop = 0):
        """
        Implementation of a Hierarchical Attention Network (HAN).
        :param vocab_size: Size of the input vocabulary
        :param embedding_size: Size of the word embedding
        :param hidden_size_words: number of hidden units for the word encoder.
        :param hidden_size_sent: number of hidden units for the sentence encoder.
        :batch_size: size of the minibatches passed to the HAN.
        :num_classes: number of output classes in the classification task.
        """
        super(HAN, self).__init__()
        self._hidden_size_words = hidden_size_words
        self._hidden_size_sent = hidden_size_sent
        self._embedding_dim = (vocab_size, embedding_size)
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._dropout_prop = dropout_prop
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        # Set up word encoder
        self._word_encoder = word_encoder(self._embedding_dim[1], self._hidden_size_words)
        # Set up sentence encoder
        self._sentence_encoder = sentence_encoder(self._hidden_size_words * 2, self._hidden_size_sent)
        # Set up a linear layer
        self._linear1 = nn.Linear(self._hidden_size_sent * 2, self._num_classes)
    def forward(self, seqs, seq_lens, hid_state_word, hid_state_sent, return_attention_weights = False):
        """
        :param batch_in: list of input documents of size batch_size input document with dim (sentence x seq_length)
        :param return_attention_weights: if True, return attention weights

        :return: tensor of shape (batch_size, num_classes) and, optionally, the attention vectors for the word and sentence encoders.
        """
        # Placeholders
        batched_sentences = None
        batch_length = len(seqs)
        # If return attention weights
        if return_attention_weights:
            word_weights = []
        # For each, do ...
        for i, seqdata in enumerate(zip(seqs,seq_lens)):
            # Unzip
            seq, seq_len = seqdata
            # Embedding
            embedded = self.embedding(seq)
            # Pack sequences
            x_packed = pack_padded_sequence(embedded, seq_len, batch_first=True, 
                                            enforce_sorted=False)
            # Word encoder
            we_out, hid_state = self._word_encoder(x_packed, hid_state_word)
            # Cat sentences together
            if batched_sentences is None:
                batched_sentences = we_out
            else:
                batched_sentences = torch.cat((batched_sentences, we_out), 0)
                # Sentence encoder
                out_sent, hid_sent = self._sentence_encoder(batched_sentences.permute(1,0,2), hid_state_sent)
            # Cat the attention weights
            if return_attention_weights:
                word_weights.append(hid_state[1].data)
                # If last sentence
                if i == batch_length:
                    sentence_weights = hid_sent[1].data
        # Apply dropout
        out_sent_dropout = F.dropout(out_sent.squeeze(0), p=self._dropout_prop)
        # Linear layer & softmax
        prediction_out = F.softmax(self._linear1(out_sent_dropout), dim = 1)
        # Return
        if return_attention_weights:
            # Compute attention weights for words and sentences
            return(prediction_out, [word_weights, sentence_weights])
        else:
            return(prediction_out)

    def init_hidden_sent(self):
        return Variable(torch.zeros(2, self._batch_size, self._hidden_size_sent))

    def init_hidden_word(self):
        return Variable(torch.zeros(2, self._batch_size, self._hidden_size_words))