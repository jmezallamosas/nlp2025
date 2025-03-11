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

def train_word2vec(data: list[list[str]], embeddings_size: int,
                    window: int = 5, min_count: int = 1, sg: int = 1) -> Word2Vec:
    """
    Create new word embeddings based on our data.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A gensim Word2Vec model
        https://radimrehurek.com/gensim/models/word2vec.html

    """

    model = Word2Vec(data, vector_size=embeddings_size, window=window, min_count=min_count, sg=sg)
    return model