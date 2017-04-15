"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file>
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle

from corpus.twitter_corpus_reader import TwitterCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargo el test corpus
    corpus = TwitterCorpusReader('../../corpus/', 'TestNiUnaMenos.txt')
    long_sent = corpus.sents()
    # Cargo el modelo entrenado.
    filename = opts['-i']
    f = open(opts['-i'], 'rb')
    model = pickle.load(f)
    print("Evaluating")
    print("."*50)
    print("Perplexity " + str(model.perplexity(long_sent)))
    print("Cross-Entrophy " + str(model.cross_entrophy(long_sent)))
    # Cierro archivo.
    f.close()
