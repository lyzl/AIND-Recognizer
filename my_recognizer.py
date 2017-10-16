import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    valid_models = {descript:model for descript,model in models.items() if model != None}

    probabilities = [get_word_probs(valid_models, *test_set.get_item_Xlengths(i)) for i,_ in enumerate(test_set.wordlist)]
    guesses = [max(word_probs.keys(), key=lambda word: word_probs[word]) for word_probs in probabilities]

    return probabilities, guesses

def get_word_probs(models, X, lengths):
  probs = {}
  for descript, model in models.items():
    try:
      probs[descript] = model.score(X, lengths)
    except:
      probs[descript] = float("-inf")
  return probs

