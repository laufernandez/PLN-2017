"""
Evaulate a language model using the test set.

Usage:
  eval.py -i <file> [-c <corpus>]
  eval.py -h | --help

Options:
  -i <file>     Language model file.
  -h --help     Show this screen.
  -c <corpus>   Corpus to test [default: politicosargentinos]:
                  politicosargentinos: Argentine Politicians Tweets.
                  niunamenos: Ni Una Menos Tweets (MicaelaGarcia).
"""

from docopt import docopt
import pickle

from corpus.twitter_corpus_reader import TwitterCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargo el corpus segun seleccion del usuario.
    c = opts['-c']
    if c == 'politicosargentinos':
        txt_file = 'TestPoliticosArgentinos.txt'
    elif c == 'niunamenos':
        txt_file = 'TestNiUnaMenos.txt'
    else:
        __doc__

    corpus = TwitterCorpusReader('../../corpus/', txt_file)

    long_sent = corpus.sents()
    # Cargo el modelo entrenado.
    filename = opts['-i']
    f = open(opts['-i'], 'rb')
    model = pickle.load(f)
    print("Evaluating")
    print("."*50)
    print("Perplexity " + str(model.perplexity(long_sent)))
    print("Cross-Entropy " + str(model.cross_entropy(long_sent)))
    # Cierro archivo.
    f.close()
