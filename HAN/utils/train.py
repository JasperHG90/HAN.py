import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

# Create function that makes a minibatch
def batcher(wiki_data, batch_size):
    """
    Create a minibatch from DocData dataset
    """
    rp = np.random.permutation(wiki_data.__len__())[:batch_size]
    # Get X, y
    batch = [wiki_data.__getitem__(idx) for idx in list(rp)]
    # Return
    return(batch)

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
            batch[j] = (batch[j][0] + (doc_len - len(batch[j][0])) * [torch.tensor([0]).type(torch.long)], batch[j][1])
    for i in range(doc_len):
        # Get sequences
        sent_seq = [b[0][i] for b in batch]
        # Record lengths of sequences
        sent_lens = [len(sent) for sent in sent_seq]
        # Create numpy
        # Pad the sequence
        sent_seq_padded = pad_sequence(sent_seq, batch_first=True, padding_value=0).to(device)
        # Append
        seq_final.append(sent_seq_padded)
        seq_lens.append(sent_lens)
    # Return
    return(seq_final, seq_lens)

# Function to split input data into train // test
def split_data(X, y, seed = None, p = 0.05):
    """
    Split data into train and test
    """
    # Create batched data
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    # Get proportion
    num_val = int(np.round(len(X) * p, 0))
    train_idx = indices[:len(X) - num_val]
    test_idx = indices[(len(X) - num_val):]
    # Split
    train_data = [X[index] for index in train_idx]
    train_label = [y[index] for index in train_idx]
    val_label = [y[index] for index in test_idx]
    val_data = [X[index] for index in test_idx]
    # Return
    return((train_data, train_label), (val_data, val_label))

# Training regime for HAN model
def train_HAN(X, y, model, optimizer, criterion, epochs = 10, 
              val_split = .1, batch_size=64, device = "cpu"):
    """
    Train a Hierarchical Attention Network
    :param X: input documents. Structured as a list of lists, where one entry is a list of input sentences.
                Note: documents can be different sizes in terms of length of sentences and number of sentences.
                        both are padded.
    :param y: numpy array containing the output labels
    :param model: a HAN model.
    :param optimizer: optimizer used for gradient descent.
    :param criterion: optimization criterion
    :param epochs: number of epochs to train the model.
    :param val_split: proportion of data points of total documents used for validation.
    :param batch_size: size of the minibatches.
    :param device: either one of 'cpu' or 'cuda' if GPU is available.
    :return: Tuple containing:
        1. Trained pytorch model
        2. Training history. Dict containing 'training_loss', 'training_acc' and 'validation_acc'
    """
    # Number of input examples
    n_examples = len(X)
    # Keep track of training loss / accuracy
    training_loss = []
    training_acc = []
    validation_loss = []
    validation_acc = []
    validation_precision = []
    validation_recall = []
    validation_f1 = []
    # For each epoch, train the mopdel
    for epoch in range(0, epochs):
        epoch += 1
        running_loss = 0.0
        running_acc = 0.0
        # Split data
        batch_train, batch_val = split_data(X, y, p = val_split)
        # Make datasets
        batch_train_data = DocData(batch_train[0], batch_train[1])
        batch_val_data = DocData(batch_val[0], batch_val[1])
        # For each train/test example
        for i in range(n_examples // batch_size):
            # Set model to train
            model.train()
            # Init the hidden states
            hid_state_word = model.init_hidden_word()
            hid_state_sent = model.init_hidden_sent()
            # Draw a batch
            current_batch = batcher(batch_train_data, batch_size)
            # Process input batches
            #  What happens here is as follows:
            #   (1) all first sentences go with first sentences for all docs etc.
            #   (2) Apply packed_sequences to make variable-batch lengths
            seqs, lens = process_batch(current_batch, device = device)
            # GT labels
            labels_ground_truth = torch.tensor([b[1] for b in current_batch]).to(device)
            # Zero gradients
            model.zero_grad()
            # Predict output
            predict_out = model(seqs, torch.tensor(lens).type(torch.long).to(device), hid_state_word.to(device), hid_state_sent.to(device))
            # Get max
            predict_class = torch.argmax(predict_out, dim=1).cpu().numpy()
            # Loss
            loss_out = criterion(predict_out, labels_ground_truth)
            # As item
            loss_value = loss_out.cpu().item()
            # GT labels to numpy
            labels_ground_truth = labels_ground_truth.cpu().numpy()
            acc_batch = sum(predict_class == labels_ground_truth) / labels_ground_truth.shape[0]
            # Update loss and accuracy
            running_loss += (loss_value - running_loss) / (i + 1)
            running_acc += (acc_batch - running_acc) / (i + 1)
            # Print if desired
            if i % 5 == 0:
                print("Loss is {} on iteration {} for epoch {} ...".format(np.round(running_loss, 3), i, epoch))
            # Produce gradients
            loss_out.backward()
            # Make step
            optimizer.step()
        # Append loss
        training_loss.append(running_loss)
        training_acc.append(running_acc)
        # On validation data
        with torch.no_grad():
            # Set model to evaluation mode
            model.eval()
            # Get batches
            io = batcher(batch_val_data, len(batch_val_data.X))
            # Init the hidden states
            hidden_state_word = model.init_hidden_word()
            hidden_state_sent = model.init_hidden_sent()
            # Process true label
            ytrue = [doc[1] for doc in io]
            ytrue = torch.tensor(ytrue).to(device)
            # Process batches
            seqs, lens = process_batch(io, device = device)
            # To outcome probabilities
            out = model(seqs, lens, hidden_state_word.to(device), hidden_state_sent.to(device))
            loss_out = criterion(out, ytrue)
            # To class labels
            out = torch.argmax(out, dim=1)
        # Make true values into numpy array
        ytrue = ytrue.cpu().numpy()
        # Metrics
        val_metrics = metrics.precision_recall_fscore_support(ytrue,
                                                              out.cpu().numpy(),
                                                              average="weighted")
        # Acc
        val_acc = np.round(sum(out.cpu().numpy() == ytrue) / ytrue.shape[0], 3)
        validation_acc.append(val_acc)
        validation_loss.append(loss_out.cpu().item())
        validation_precision.append(val_metrics[1])
        validation_recall.append(val_metrics[2])
        validation_f1.append(val_metrics[0])
        # Print
        print("-------------")
        print("Training Loss is {} at epoch {} ...".format(np.round(running_loss, 3), epoch))
        print("Training accuracy is {} at epoch {} ...".format(np.round(running_acc, 3), epoch))
        print("Validation accuracy is {} at epoch {} ...".format(val_acc, epoch))
        print("-------------")
    # Return
    return(model, {"training_loss": np.round(training_loss, 3),
                   "training_accuracy": np.round(training_acc, 2),
                   "validation_loss":np.round(validation_loss, 3),
                   "validation_accuracy": validation_acc,
                   "validation_precision":np.round(validation_precision, 2),
                   "validation_recall":np.round(validation_recall, 2),
                   "validation_f1":np.round(validation_f1, 2)})

# Create a dataset to hold both the documents and the labels
class DocData(Dataset):
    def __init__(self, X, y):
        # Must be same length
        assert len(X) == len(y), "'X' and 'y' different lengths"
        self.X = X
        self.y = y
        self.len = len(X)
    def __getitem__(self, index):
        # Retrieve X
        X = [torch.tensor(sent).type(torch.long) for sent in self.X[index]]
        # Each sentence to tensor
        return((X, 
                torch.tensor(self.y[index]).type(torch.long)))
    def __len__(self):
        return(self.len)