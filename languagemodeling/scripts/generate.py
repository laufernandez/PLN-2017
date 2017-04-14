"""
Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.
 """

from docopt import docopt
import pickle

from languagemodeling.ngram import NGramGenerator

if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargo el modelo desde el archivo.
    filename = opts['-i']
    f = open(opts['-i'], 'rb')
    n = int(opts['-n'])
    model = pickle.load(f)
    # Creo el ngrama.
    ngram = NGramGenerator(model)
    print("Caso " + str(ngram.n) + "-grama")
    # Imprimo sentencia por sentencia.
    for i in range(n):
        print("Sentencia " + str(i + 1) + (":"))
        sent = ' '.join(ngram.generate_sent())
        print(sent)
    # Cierro archivo.
    f.close()
