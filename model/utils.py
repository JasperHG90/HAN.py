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
