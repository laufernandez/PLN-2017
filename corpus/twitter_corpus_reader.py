from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import RegexpTokenizer

class TwitterCorpusReader(PlaintextCorpusReader):
    """
    Corpus Reader personalizado para el tokenizado de tweets.
    """

    def __init__(self, root, fileids):
        """
        Construye un nuevo corpus reader personalizado para el tokenizado
        correcto de tweets. Tiene en cuenta el uso de hashtags, el formato
        de nombre de usuarios y los links compartidos.
        :param root: directorio donde se encuentra el corpus.
        :param fileids: archivo(s) que forman el corpus.
        """

        self._pattern = r'''(?ix)               # Flag
                    (?:(?:[Hh][Tt][Tt][Pp][Ss]?:\/\/)|[wW][Ww][Ww])
                    (?:\/?\.?\d?[a-zA-Z]?)+     # Urls completas.
                    |(?:[A-Z]\.)+               # Abreviaciones, e.g. U.S.A.
                    | (?:[Ss]r\.|[Ss]ra\.)      # Sr. Sra. sr. sra.
                    | (?:[Dd]r\.|[Dd]ra\.)      # Dr. Dra. dr. dra.
                    | \.\.\.                    # Puntos suspensivos.
                    | \@\w+(?:-\w+)*            # Nombres de usuario de Twitter.
                    | \d+(?:[\.\,]\d+)%         # Porcentajes
                    | \#?\w+(?:-\w+)*           # Palabras/Hashtags.
                    | \$?\d+(?:[\.\,]\d+)?      # Numeros decimales, precios.
                    | [][.,;"'?():-_`…]         # Tokens especiales.
                    '''

        self._tokenizer = RegexpTokenizer(self._pattern)

        PlaintextCorpusReader.__init__(self, root, fileids, word_tokenizer=self._tokenizer)

