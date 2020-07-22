# Predict HAN outputs
from HAN.utils.train import process_batch, DocData
import torch
import numpy as np

def predict_HAN(model, snippets, labels, batch_size = 128, return_probabilities = False, return_attention = False, device = "cpu"):
    """
    Create predictions for a HAN
    :param model: HAN model
    :param snippets: list of vectorized input texts
    :param labels: labels belonging to the snippets
    :param batch_size: size of the input batches to the model
    :param device: device on which the model is run
    :return: tuple containing predictions and ground truth labels
    """
    assert len(snippets) == len(labels), "Snippets and labels are of different lengths ..."
    # Turn into docdataset
    dataset = DocData(snippets, labels)
    n = len(snippets)
    total = n // batch_size
    remainder = n % batch_size
    # Make indices
    idx = []
    start_idx = 0
    for batch_idx in range(1, total+1):
        idx.append((start_idx, batch_idx * batch_size))
        start_idx += batch_size
    # If remainder
    if remainder > 0:
        idx.append((start_idx, start_idx + remainder))
    # For each pair, predict
    predictions = []
    ground_truth = []
    probs_pred = []
    for start_idx, stop_idx in idx:
        # Get batch
        inbatch = [dataset.__getitem__(idx) for idx in range(start_idx, stop_idx)]
        # Process batch
        seqs, lens = process_batch(inbatch, device = device)
        # Init hidden states
        hidden_state_word = model.init_hidden_word().to(device)
        hidden_state_sent = model.init_hidden_sent().to(device)
        # Predict
        with torch.no_grad():
            model.eval()
            if return_attention:
                probs, attn = model(seqs, lens, hidden_state_word, hidden_state_sent, return_attention_weights=True)
            else:
                probs = model(seqs, lens, hidden_state_word, hidden_state_sent, return_attention_weights=False)
        # To classes
        out = torch.argmax(probs, dim=1).cpu().numpy()
        # Get true label
        ytrue = [batch[1] for batch in inbatch]
        ytrue = torch.tensor(ytrue).cpu().numpy()
        # Cat together
        predictions.append(out)
        ground_truth.append(ytrue)
        probs_pred.append(probs.cpu().numpy())
    # Stack predictions & ground truth
    if not return_probabilities and not return_attention:
        return(np.hstack(predictions), np.hstack(ground_truth))
    elif return_probabilities:
        return(np.hstack(predictions), np.concatenate(probs_pred, axis=0), np.hstack(ground_truth))
    else:
        return(np.hstack(predictions), attn, np.hstack(ground_truth))