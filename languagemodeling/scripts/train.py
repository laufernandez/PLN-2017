"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file>
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -o <file>     Output model file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
from languagemodeling.ngram import NGram, AddOneNGram
# Importo mi corpus reader personalizado.
from corpus.twitter_corpus_reader import TwitterCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # load the data
    corpus = TwitterCorpusReader('../../corpus/', 'NiUnaMenos.txt')
    sents = corpus.sents()

    # train the model
    n = int(opts['-n'])
    # Si pasa algun argumento como opcion para el modelo.
    m = opts['-m']  # Por default es 'ngram'.
    if m == 'addone':
        print('Training an AddOneNgram Model')
        model = AddOneNGram(n, sents)
    elif m == 'ngram':
        print('Training an Ngram Model')
        model = NGram(n, sents)
    else:
        __doc__
    # save it
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    print('.' * 50)
    print('Trained!')
    f.close()
