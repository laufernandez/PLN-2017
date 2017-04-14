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

from languagemodeling.ngram import NGramGenerator, AddOneNGram

if __name__ == '__main__':
    opts = docopt(__doc__)

    # Cargo el modelo desde el archivo.
    filename = opts['-i']
    f = open(opts['-i'], 'rb')
    n = int(opts['-n'])
    model = pickle.load(f)
    if type(model) != AddOneNGram:
        # Creo el ngrama.
        ngram = NGramGenerator(model)
        print("Caso " + str(ngram.n) + "-grama")
        # Imprimo sentencia por sentencia.
        for i in range(n):
            print("Sentencia " + str(i + 1) + (":"))
            sent = ' '.join(ngram.generate_sent())
            print(sent)
    # Para el caso de un modelo AddOneNGram, el metodo de generacion
    # de las probabilidades condicionales falla. No se suman los counts para
    # las combinaciones que no aparecen en el corpus, pero si se dividen por V
    # a las que si aparecen.
    # De esta manera, la variable que guarda la distribucion acumulada nunca
    # alcanza a cubrir el rango total de la distribucion uniforme (0, 1).
    else:
        print('This generator doesn\'t work for an AddOneNGram model!')
    # Cierro archivo.
    f.close()
