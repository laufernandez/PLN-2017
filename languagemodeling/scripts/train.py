"""
Train an n-gram model.

Usage:
  train.py -n <n> [-m <model>] -o <file> [-c <corpus>]
  train.py -h | --help

Options:
  -n <n>        Order of the model.
  -m <model>    Model to use [default: ngram]:
                  ngram: Unsmoothed n-grams.
                  addone: N-grams with add-one smoothing.
  -o <file>     Output model file.
  -c <corpus>   Corpus to use [default: politicosargentinos]:
                  politicosargentinos: Argentine Politicians Tweets.
                  niunamenos: Ni Una Menos Tweets (MicaelaGarcia).
  -h --help     Show this screen.
"""

from docopt import docopt
import pickle
from languagemodeling.ngram import NGram, AddOneNGram
# Importo mi corpus reader personalizado.
from corpus.twitter_corpus_reader import TwitterCorpusReader


if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargo el corpus segun seleccion del usuario.
    c = opts['-c']
    if c == 'politicosargentinos':
        txt_file = 'PoliticosArgentinos.txt'
    elif c == 'niunamenos':
        txt_file = 'NiUnaMenos.txt'
    else:
        __doc__
    corpus = TwitterCorpusReader('../../corpus/', txt_file)

    # Entreno el modelo.
    sents = corpus.sents()
    n = int(opts['-n'])  # Orden
    m = opts['-m']  # Por default es 'ngram'.
    if m == 'addone':
        print('Training an AddOneN{}-gram Model'.format(n))
        model = AddOneNGram(n, sents)
    elif m == 'ngram':
        print('Training an N{}-gram Model'.format(n))
        model = NGram(n, sents)
    else:
        __doc__

    # Guardo los modelos en archivos binarios.
    filename = opts['-o']
    f = open(filename, 'wb')
    pickle.dump(model, f)
    print('.' * 50)
    print('Trained!')
    f.close()
