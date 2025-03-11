# Rename this file to neurallm_utils.py!

# for word tokenization
import nltk
import csv
import torch
import torch.nn as nn
from gensim.models import Word2Vec

nltk.download('punkt')
nltk.download('punkt_tab')

SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"


# PROVIDED
def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   space_char: str = ' ',
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    space_char (str): if by_char is True, use this character to separate to replace spaces
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  inner_pieces = None
  if by_char:
    line = line.replace(' ', space_char)
    inner_pieces = list(line)
  else:
    # otherwise use nltk's word tokenizer
    inner_pieces = nltk.word_tokenize(line)

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  # always count the unigrams
  return tokens


# PROVIDED
def read_file_spooky(datapath: str, ngram: int, by_character:bool = False) -> list:
    '''Reads and Returns the "data" as list of list (as shown above)'''
    data = []
    with open(datapath, encoding= 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # THIS IS WHERE WE GET CHARACTERS INSTEAD OF WORDS
            # replace spaces with underscores
            data.append(tokenize_line(row['text'].lower(), ngram, by_char=by_character, space_char="_"))
    return data

# PROVIDED
def get_file_authors(datapath: str) -> list:
    '''Reads and Returns the "data" as list of list (as shown above)'''
    data = []
    with open(datapath, encoding= 'utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row['author'])
    return data


def save_pytorch(obj, filename: str) -> None:
    """
    Saves a PyTorch object to a file.

    Params:
        obj: The object.
        filename: The destination file.
    """
    torch.jit.script(obj).save(filename)

def load_pytorch(filename: str):
    """
    Saves a PyTorch object to a file.

    Params:
        filename: The saved model file.
    """
    model = torch.jit.load(filename)
    # Set the model to evaluation mode
    model.eval()
    return model

# PROVIDED
def save_word2vec(embeddings: Word2Vec, filename: str) -> None:
    """
    Saves weights of trained gensim Word2Vec model to a file.

    Params:
        obj: The object.
        filename: The destination file.
    """
    embeddings.save(filename)

# PROVIDED
def load_word2vec(filename: str) -> Word2Vec:
    """
    Loads weights of trained gensim Word2Vec model from a file.

    Params:
        filename: The saved model file.
    """
    return Word2Vec.load(filename)


# You will put your create_embedder function here once you finish Task 2 so that you
# can use it in the other tasks easily.

# NOT PROVIDED
# After you are happy with this function (Task 2), copy + paste it into the bottom of 
# your neurallm_utils.py file
# You'll need it for the next task!
def create_embedder(raw_embeddings: Word2Vec) -> torch.nn.Embedding:
    """
    Create a PyTorch embedding layer based on our data.

    We will *first* train a Word2Vec model on our data.
    Then, we'll use these weights to create a PyTorch embedding layer.
        `nn.Embedding.from_pretrained(weights)`


    PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
    Gensim Word2Vec docs: https://radimrehurek.com/gensim/models/word2vec.html

    Pay particular attention to the *types* of the weights and the types required by PyTorch.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A PyTorch embedding layer
    """

    # Hint:
    # For later tasks, we'll need two mappings: One from token to index, and one from index to tokens.
    # It might be a good idea to store these as properties of your embedder.
    # e.g. `embedder.token_to_index = ...`

    # Create mappings
    
    #get word vectors
    word_vectors = raw_embeddings.wv.vectors  
    #convert to tensor 
    wv_tensor = torch.tensor(word_vectors, dtype=torch.float32)
    #pass in new weights  
    embedding = torch.nn.Embedding.from_pretrained(wv_tensor)

    token_to_index = dict()
    index_to_token = dict()
    for token in raw_embeddings.wv.index_to_key:
        token_to_index[token] = raw_embeddings.wv.key_to_index[token]
        index_to_token[raw_embeddings.wv.key_to_index[token]] = token

    embedding.token_to_index = token_to_index
    embedding.index_to_token = index_to_token
    #return embedding
    return embedding

# -------------------------------
# Data processing functions
# -------------------------------

def encode_tokens(data: list[list[str]], embedder: torch.nn.Embedding) -> list[list[int]]:
    """
    Replaces each natural-language token with its embedder index.

    e.g. [["<s>", "once", "upon", "a", "time"],
          ["there", "was", "a", ]]
        ->
        [[0, 59, 203, 1, 126],
         [26, 15, 1]]
        (The indices are arbitrary, as they are dependent on your embedder)

    Params:
        data: The corpus
        embedder: An embedder trained on the given data.
    """

    finalList = []
    for list in data:
        currList = []
        for word in list:
            index = embedder.token_to_index[word]
            currList.append(index)
        finalList.append(currList)

    return finalList


def create_ngrams(tokens: list, n: int) -> list:
    """Creates n-grams for the given token sequence.
    Args:
      tokens (list): a list of tokens as strings
      n (int): the length of n-grams to create

    Returns:
      list: list of tuples of strings, each tuple being one of the individual n-grams
    """
    # STUDENTS IMPLEMENT
    res = []
    for i in range(0, len(tokens)-n):
        #append n gram + yth value
        res.append(tokens[i:i+n+1])
    return res

def generate_ngram_training_samples(encoded: list[list[int]], ngram: int) -> list:
    """
    Takes the **encoded** data (list of lists of ints) and 
    generates the training samples out of it.
    
    Parameters:
        up to you, we've put in what we used
        but you can add/remove as needed
    return: 
    list of lists in the format [[x1, x2, ... , x(n-1), y], ...]
    """

    #1 2 3 4
    #[1,2, y=3]
    #[2,3, y=4]

    # if you'd like to use tqdm, you can use it like this:
    # for i in tqdm(range(len(encoded))):
    final_list = []
    for list in encoded:
        currList = create_ngrams(list, ngram-1)
        final_list.extend(currList)
    return final_list

def split_sequences(training_sample):
    x_sample = []
    y_sample = []
    for line in training_sample:
        x_sample.append(line[0:-1])
        y_sample.append(line[-1])
    return x_sample, y_sample

def create_dataloaders(X: list, y: list, num_sequences_per_batch: int, 
                       test_pct: float = 0.1, shuffle: bool = True) -> tuple[torch.utils.data.DataLoader]:
    """
    Convert our data into a PyTorch DataLoader.    
    A DataLoader is an object that splits the dataset into batches for training.
    PyTorch docs: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        https://pytorch.org/docs/stable/data.html

    Note that you have to first convert your data into a PyTorch DataSet.
    You DO NOT have to implement this yourself, instead you should use a TensorDataset.

    You are in charge of splitting the data into train and test sets based on the given
    test_pct. There are several functions you can use to acheive this!

    The shuffle parameter refers to shuffling the data *in the loader* (look at the docs),
    not whether or not to shuffle the data before splitting it into train and test sets.
    (don't shuffle before splitting)

    Params:
        X: A list of input sequences
        Y: A list of labels
        num_sequences_per_batch: Batch size
        test_pct: The proportion of samples to use in the test set.
        shuffle: INSTRUCTORS ONLY

    Returns:
        One DataLoader for training, and one for testing.
    """
    
    dataSet = TensorDataset(torch.tensor(X), torch.tensor(y))
    test_size = int(len(dataSet)*test_pct)
    train_size = len(dataSet) - test_size
    train_data, test_data = torch.utils.data.random_split(dataSet, [train_size, test_size])
    dataloader_train = DataLoader(train_data, batch_size=num_sequences_per_batch, shuffle=shuffle)
    dataloader_test = DataLoader(test_data, batch_size=num_sequences_per_batch, shuffle=shuffle)
    return dataloader_train, dataloader_test

# -------------------------------
# FFNN Model and Training Functions
# -------------------------------

class FFNN(nn.Module):
    """
    A Feed-Forward Neural Network for language modeling.
    """
    def __init__(self, vocab_size: int, ngram: int, embedding_layer: torch.nn.Embedding, hidden_units=128):
        """
        Initialize a new untrained model.
        
        Params:
            vocab_size: Number of words in the vocabulary.
            ngram: The N value (window size) for training.
            embedding_layer: Pre-trained embedding layer.
            hidden_units: Number of hidden units in the hidden layer.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.ngram = ngram
        self.embedding_layer = embedding_layer
        self.hidden_units = hidden_units
        
        # Get embedding dimension from the provided embedder.
        embedding_size = embedding_layer.embedding_dim
        
        # Define the network: flatten embedded n-gram tokens, then two linear layers with ReLU.
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=(ngram-1) * embedding_size, out_features=hidden_units, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=vocab_size, bias=True)
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Params:
            X: Tensor of input indices with shape (batch_size, ngram-1)
        
        Returns:
            Logits of shape (batch_size, vocab_size).
        """
        embedded = self.embedding_layer(X)
        flat_embedded = self.flatten(embedded)
        logits = self.linear_relu_stack(flat_embedded)
        return logits

def train_one_epoch(dataloader, model, optimizer, loss_fn):
    epoch_loss = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()                  # Zero gradients for this batch.
        outputs = model(inputs)                # Forward pass.
        batch_loss = loss_fn(outputs, labels)  # Compute loss.
        batch_loss.backward()                  # Backpropagation.
        optimizer.step()                       # Update weights.
        epoch_loss += batch_loss.item()
    return epoch_loss

def train(dataloader, model, epochs: int = 1, lr: float = 0.001) -> None:
    """
    Train the model.
    
    Params:
        dataloader: Training data loader.
        model: The model to train.
        epochs: Number of epochs.
        lr: Learning rate.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n_batches = len(dataloader)
    
    model.train()  # Set the model to training mode.
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = train_one_epoch(dataloader, model, optimizer, loss_fn)
        avg_epoch_loss = epoch_loss / n_batches
        print(f"Epoch: {epoch}, Average Loss: {avg_epoch_loss:.4f}")
        # Log metrics to wandb
        wandb.log({"epoch": epoch, "avg_epoch_loss": avg_epoch_loss})

def full_pipeline(data, word_embeddings_filename: str, 
                  batch_size: int,
                  ngram: int,
                  hidden_units: int = 128,
                  epochs: int = 1,
                  lr: float = 0.001,
                  test_pct: float = 0.1) -> FFNN:
    """
    Run the full training pipeline from loading embeddings to model training.
    
    Params:
        data: Raw data as a list of lists of tokens (here, integer indices).
        word_embeddings_filename: Filename for the pre-trained embeddings.
        batch_size: Batch size for training.
        ngram: N-gram size.
        hidden_units: Number of hidden units.
        epochs: Number of epochs.
        lr: Learning rate.
        test_pct: Percentage of data for testing (not used in training).
    
    Returns:
        The trained FFNN model.
    """
    # Load embeddings and create an embedder.
    token_embeddings = load_word2vec(word_embeddings_filename)
    embedder = create_embedder(token_embeddings)
    
    # Preprocess data.
    encoded_tokens = encode_tokens(data, embedder)
    vocab_size = embedder.num_embeddings
    training_sample = generate_ngram_training_samples(encoded_tokens, ngram)
    x_sample, y_sample = split_sequences(training_sample)
    dataloader_train, _ = create_dataloaders(x_sample, y_sample, batch_size, test_pct)
    
    # Initialize the model.
    model = FFNN(vocab_size=vocab_size, ngram=ngram, embedding_layer=embedder, hidden_units=hidden_units)
    
    # Train the model.
    train(dataloader=dataloader_train, model=model, epochs=epochs, lr=lr)
    
    return model

# -------------------------------
# Prediction and generation functions
# -------------------------------

# Create a function that predicts the next token in a sequence.
def predict(model, input_tokens) -> str:
    """
    Get the model's next word prediction for an input.
    This is where you'll use the softmax function!
    Assume that the input tokens do not contain any unknown tokens.

    Params:
        model: Your trained model
        input_tokens: A list of natural-language tokens. Must be length N-1.

    Returns:
        The predicted token (not the predicted index!)
    """
    # YOUR CODE HERE
	# Encode tokens
    encoded_tokens = [model.embedding_layer.token_to_index[token] for token in input_tokens]
    
	# Trasform to tensor
    encoded_tokens = torch.tensor([encoded_tokens]) # Dim [1, ngram-1]
    
    # Setting model to evaluation mode turns off Dropout and BatchNorm making the predictions deterministic
    model.eval()  # Set the model to evaluation mode if you haven't already
    
    with torch.no_grad(): # Speeds up inference and reduces memory usage by not having to calcualte gradients
        logits = model(encoded_tokens) # Forward pass on the model
        probability = nn.functional.softmax(logits, dim=1) # Normalize z scores to probability
        predicted_idx = torch.multinomial(probability, num_samples=1).item()

        #predicted_idx = probability.argmax(dim=1).item() # Retrieve int value
		
	# Transform index to natural-language token
    predicted_token = model.embedding_layer.index_to_token[predicted_idx] 
    
    return predicted_token

from typing import List
# Generate a sequence from the model until you get an end of sentence token.
def generate(model, seed: List[str], max_tokens: int = None) -> List[str]:
    """
    Use the trained model to generate a sentence.
    This should be somewhat similar to generation for HW2...
    Make sure to use your predict function!

    Params:
        model: Your trained model
        seed: [w_1, w_2, ..., w_(n-1)].
        max_tokens: The maximum number of tokens to generate. When None, should gener
            generate until the end of sentence token is reached.

    Return:
        A list of generated tokens.
    """ 
    n_tokens = 0 # Count tokens that have been generated
    tokens = seed.copy() # Copy of initial seed
    end_token = "<\s>"
    
    while True:
        for_prediction = seed[-(model.ngram-1):]
        predicted_token = predict(model, for_prediction)
        if predicted_token == end_token:
        	break
        tokens.append(predicted_token)
        n_tokens += 1
        if max_tokens is not None and n_tokens >= max_tokens:
            break
        
    return tokens

def generate_sentences(model, seed: List[str],  n_sentences: int, max_tokens: int = None) -> List[str]:
    return [generate(model, seed, max_tokens) for i in range(n_sentences)]

# you might want to define some functions to help you format the text nicely
# and/or generate multiple sequences

def format_sentence(tokens_list: List[List[str]], by_char = False) -> str:
  """Removes <s> at the start of the sentence and </s> at ehe end. Joins the list of tokens into a string and capitalizes it.
  Args:
    tokens (list(list)): the list of tokens list to be formatted into a sentence

  Returns:
    string: formatted sentence as a string
  
  """
  text = "" # Initializing final sentence
  for tokens in tokens_list: # Parsing through each individual sentence
    while tokens[0] == '<s>': # Removes all <s> at the beggining even if there are several for ngram > 2 models
      tokens.pop(0)
    if tokens[-1] == '</s>': # Removes the one </s> at the end of the sentence
      tokens.pop(-1)
    if by_char:
      sentence = "".join(tokens) # Converts list of tokens into a string
      sentence = sentence.capitalize() # Capitalizes the first letter of each sentence
    else:
      sentence = " ".join(tokens) # Converts list of tokens into a string
      sentence = sentence.capitalize() # Capitalizes the first letter of each sentence
    text += sentence + ".\n" # Adds a period and space separator between sentences
  return text.strip(" ") # Removes the last space in the last sentence
