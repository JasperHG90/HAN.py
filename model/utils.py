"""
HAN utility functions:
    To make the HAN self-contained, I put all utility functions in this python file. The preprocessing
    steps and dataset construction are a little different from the other models. The preprocessing 
    functions are as follows:
        1. Embedding_FastText: creates a Pytorch embedding layer from pre-trained weights
        2. WikiDocData: Pytorch Dataset used to store & retrieve wikipedia data.
        3. batcher: function that creates minibatches for training
        4. process_batch: processes a minibatch of wikipedia articles.
        5. split_data: function that splits wikipedia data into train & test
        6. train_han: training regime for the HAN.
"""

# Create FastText embedding for PyTorch
def Embedding_FastText(weights, freeze_layer = True):
    """Set up a pytorch embedding matrix"""
    examples, embedding_dim = weights.shape
    # Set up layer
    embedding = nn.Embedding(examples, embedding_dim)
    # Add weights
    embedding.load_state_dict({"weight": weights})
    # If not trainable, set option
    if freeze_layer:
        embedding.weight.requires_grad = False
    # Return
    return(embedding)

# Create a dataset to hold both the documents and the labels
class WikiDocData(Dataset):
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

# Create function that makes a minibatch
def batcher(wiki_data, batch_size):
    """
    Create a minibatch from WikiDocData dataset
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
def train_han(X, y, model, optimizer, criterion, epochs = 10, 
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
        batch_train_data = WikiDocData(batch_train[0], batch_train[1])
        batch_val_data = WikiDocData(batch_val[0], batch_val[1])
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
    return(model, {"training_loss": training_loss,
                   "training_accuracy": training_acc,
                   "validation_loss":validation_loss,
                   "validation_accuracy": validation_acc,
                   "validation_precision":validation_precision,
                   "validation_recall":validation_recall,
                   "validation_f1":validation_f1})

def predict_HAN(model, dataset, batch_size = 128, return_probabilities = False, return_attention = False, device = "cpu"):
    """
    Create predictions for a HAN

    :param model: HAN model
    :param dataset: WikiDocData dataset
    :param batch_size: size of the input batches to the model
    :param device: device on which the model is run
    :return: tuple containing predictions and ground truth labels
    """
    n = len(dataset.X)
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

