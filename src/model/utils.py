# Utility functions for the HAN
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

# Function to process a batch
def process_batch(batch, device = "cpu"):
    """
    Process a minibatch for handing off to the HAN
    """
    # Get the length of a document in the batch
    doc_len = np.max([len(b[0]) for b in batch])
    # Place the first sentences for each doc in one list, second sentences also etc.
    seq_final = []
    seq_lens = []
    # Pad documents with fewer sentences than the maximum number of sequences
    # This allows training of documents of different size
    for j in range(len(batch)):
        if len(batch[j][0]) < doc_len:
            batch[j] = (batch[j][0] + (doc_len - len(batch[j][0])) * [[0]], batch[j][1])
    for i in range(doc_len):
        # Get sequences
        sent_seq = [torch.tensor(b[0][i]).type(torch.long).to(device) for b in batch]
        # Record lengths of sequences
        sent_lens = [len(sent) for sent in sent_seq]
        # Create numpy
        # Pad the sequence
        sent_seq_padded = pad_sequence(sent_seq, batch_first=True, padding_value=0).to(device)
        # Append
        seq_final.append(sent_seq_padded)
        seq_lens.append(sent_lens)
    # Return
    return(seq_final, seq_lens, torch.tensor([b[1] for b in batch]).to(device))

def word_attention(attention_vector, seq, idx_to_word):
    """
    Compute attention weights for each word in the sentence
    
    :param attention_vector: tensor of shape (sentence_length, word_hidden_dim)
    :param seq: the vectorized sequence of words
    :param idx_to_word: dict that maps sequence integers to words
    
    :return: dictionary where keys are the words in the sequence and value is the attention weight
    """
    # Sequence length
    seq = np.array(seq)
    seq_len = seq.shape[0]
    # Sum across hidden dimension (last axis)
    attention_summed = attention_vector.sum(axis=-1)
    # Subset
    attention_summed = attention_summed[:seq_len]
    # Normalize
    attention_normed = list(np.round(attention_summed / np.sum(attention_summed), 4))
    # Store
    return({idx_to_word[int(seq[idx])]:attention_normed[idx] for idx in range(seq_len)})

def sentence_attention(attention_vector):
    """
    Compute attention weights for each sentence
    :param attention_vector: tensor of shape (examples, sentences, sentence_hidden_dim)
    :return: dictionary where keys are sentence indices and values are sentence attention weights
    """
    # Create weights for each sample
    sent_weight = attention_vector.sum(axis=-1)
    # Normalize
    sent_weight /= sent_weight.sum()
    # To dict & return
    return({k:np.round(float(list(sent_weight)[k]), 3) for k in range(0, sent_weight.shape[0])})
