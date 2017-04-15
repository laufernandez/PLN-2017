# https://docs.python.org/3/library/collections.html
from collections import defaultdict, OrderedDict
from math import log
from random import random

BEGIN = '<s>'
END = '</s>'


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.v = 0  # Tamano del vocabulario.

        # Set de wordtypes (incluye </s>).
        _vocabulary = set()

        for sent in sents:
            # Delimitadores de inicio y fin de sentencia.
            self._delimiters(sent, n)
            # Creo el diccionario de n-gramas y (n-1)-gramas y sus frecuencias.
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                # Agrego los wordtypes al set.
                for token in ngram:
                    if token != BEGIN:
                        _vocabulary.add(token)
                # Incremento la frecuencia del n-grama.
                counts[ngram] += 1
                # Incrementa frecuencia del (n-1)-grama.
                counts[ngram[:-1]] += 1

        # Necesario para el ejercicio de suavizado Add-One.
        self.v = len(_vocabulary)  # Cantidad de wordtypes.

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram or (n-1)-gram tuple.
        """
        n = self.n
        # Chequeo n-uplas o (n-1)-uplas.
        assert len(tokens) in [n, n - 1]
        # Frecuencia asociada a la tupla que pasa como argumento.
        return self.counts[tokens]

    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        # De Tokens a Tuplas.
        token = tuple([token])
        if not prev_tokens:  # Caso unigrama (NoneType).
            assert n == 1
            prev_tokens = []
        assert len(prev_tokens) == n - 1
        prev_tokens = tuple(prev_tokens)
        # Tupla que forma el n-grama.
        ngram = prev_tokens + token
        # Probabilidades y regla de la probabilidad condicional.
        ngram_prob = float(self.count(ngram))
        prev_tokens_prob = float(self.count(prev_tokens))
        try:
            conditional_prob = ngram_prob / prev_tokens_prob
        except ZeroDivisionError:
            # Cuando prev_tokens_prob es 0.
            conditional_prob = 0

        return conditional_prob

    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.

        sent -- the sentence as a list of tokens.
        """
        n = self.n
        # Delimitadores.
        self._delimiters(sent, n)

        prob = 1
        # Recorro cada n-grama en busca de sus probabilidades.
        for i in range(len(sent) - n + 1):
            prev_tokens = sent[i: i + n - 1]  # (n-1)grama de Tk previos.
            token = sent[i + n - 1]  # Token i.
            # Markov Assumption
            prob *= self.cond_prob(token, prev_tokens)

        return prob

    def sent_log_prob(self, sent):
        """Log-probability of a sentence.

        sent -- the sentence as a list of tokens.
        """
        n = self.n
        # Delimitadores.
        self._delimiters(sent, n)

        prob = 0
        for i in range(len(sent) - n + 1):
            prev_tokens = sent[i: i + n - 1]
            token = sent[i + n - 1]
            # Propiedad del logaritmo + Markov Assumption
            try:
                # Log base 2
                prob += log(self.cond_prob(token, prev_tokens), 2)
            # Cuando cond_prob es 0.
            except ValueError:
                prob = float('-inf')

        return prob

    def _delimiters(self, sent, n):
        """Add delimiters to a sentence.

        BEGIN = <s> -- beg delimiters.
        END = </s> -- end delimiters.
        """
        # Agrego n-1 delimitadores de inicio y uno de fin de sentencia
        sent[0:0] = (n-1) * [BEGIN]
        sent.append(END)

    def perplexity(self, sents):
        """Perplexity of a language model.

        sents -- abstraction of a 'long sentence' formed by the sents in the
        test data corpus.
        """
        # Obtengo la perplexity a partir de su cross_entrophy
        perplexity = pow(2, self.cross_entrophy(sents))

        return perplexity

    def cross_entrophy(self, sents):
        """Aproximated Cross-entrophy of a language model.

        sents -- abstraction of a 'long sentence'.
        """
        log_prob = self.log_probability(sents)
        # La entropia cruzada entre la distribucion de un modelo y la
        # distribucion de palabras en un corpus dado se puede aproximar
        # como la (-)log_prob de una secuencia larga normalizada por
        # el total de palabras del corpus.
        c_entrophy = float(-1 / self.m) * log_prob

        return c_entrophy

    def log_probability(self, sents):
        """Log-probability of a very long sentence. (n-gram method).

        sents -- abstraction of a 'long sentence'.
        """
        log_prob = 0
        long_sent = []
        self.m = 0  # Total de palabras del test corpus.
        for sent in sents:
            # Por el metodo de Markov para un n-grama las probabilidades
            # de cada palabra son condicionadas por las n-1 anteriores.
            # Acumulo las palabras en una nueva sentencia larga, logrando
            # una mejor aproximacion de la log_prob total que sumando los
            # valores de cada sentencia por separado.
            for word in sent:
                self.m += 1
                long_sent.append(word)
        # Calculo la log_prob de la secuencia larga.
        log_prob += self.sent_log_prob(long_sent)

        return log_prob


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """
        self.model = model
        n = self.n = model.n

        # Diccionario de probabilidades de la forma:
        # {(prev_tks1): {(tk1):tk1_prob,...,(tkn):tkn_prob},...,(prev_tksm)...}
        self.probs = defaultdict(lambda: defaultdict(int))  # Dict de Dicts.

        # Diccionario de probabilidades ordenadas de la forma:
        # {(prev_tks1): [(tk1, tk1_prob),...,(tkm, tkm_ptob)],..,(prev_tksm)..}
        self.sorted_probs = defaultdict(OrderedDict)

        # Diccionario counts filtrado por claves de largo n.
        _ncounts = {ngram: count for ngram, count in
                    self.model.counts.items() if len(ngram) == n}

        # Inicializo el diccionario de probabilidades.
        for ngram in _ncounts:
            prev_tokens = ngram[:-1]  # Tupla de tokens previos.
            token = ngram[- 1]  # String que representa al token.
            # Probabilidad condicional del token.
            cond_prob_token = self.model.cond_prob(token, prev_tokens)
            self.probs[prev_tokens][token] = cond_prob_token

        # Inicializo probabilidades ordenadas de mayor a menor.
        for prev_tk in self.probs:
            self.sorted_probs[prev_tk] = sorted(self.probs[prev_tk].items(),
                                                key=lambda x: (-x[1], x[0]))

    def generate_sent(self):
        """Randomly generate a sentence."""
        n = self.n
        sent = []
        # Para mi primer token, (n > 1) los previos son los delimitadores <s>.
        prev_tokens = tuple((n-1) * [BEGIN])
        # Unigrama no tiene delimitadores previos.
        if n == 1:
            prev_tokens = ()
        # Agrega tokens hasta llegar al delimitador de fin.
        while(True):
            token = self.generate_token(prev_tokens)
            if token == END:
                break
            sent.append(token)
            # Genera token a partir de (n-1) tokens previos.
            prev_tokens = prev_tokens[1:] + tuple([token])
            if n == 1:
                prev_tokens = ()

        return sent

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        # Unigrama
        if not prev_tokens:
            assert n == 1
            prev_tokens = ()

        # Random con distribucion pseudo-uniforme (0,1).
        rand = random()
        # Distribucion acumulada a partir de las probabiliades de cada token.
        cum_distribution = 0

        # Algoritmo de sampleo para una multinomial.
        for token, token_prob in self.sorted_probs[prev_tokens]:
            cum_distribution += token_prob
            # Si el random cae adentro de la acumulada, elijo ese token.
            if rand <= cum_distribution:
                return token
        raise(AssertionError('Cumulative Distribution < Rand'))


class AddOneNGram(NGram):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.v = 0  # Tamano del vocabulario.

        # Set de wordtypes (incluye </s>).
        _vocabulary = set()

        for sent in sents:
            # Delimitadores de inicio y fin de sentencia.
            self._delimiters(sent, n)
            # Creo el diccionario de n-gramas y (n-1)-gramas y sus frecuencias.
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                # Agrego los wordtypes al set.
                for token in ngram:
                    if token != BEGIN:
                        _vocabulary.add(token)
                # Incremento la frecuencia del n-grama.
                counts[ngram] += 1
                # Incrementa frecuencia del (n-1)-grama.
                counts[ngram[:-1]] += 1

        # Necesario para el ejercicio de suavizado Add-One.
        self.v = len(_vocabulary)  # Cantidad de wordtypes.

    def V(self):
        """Size of the vocabulary.
        """
        return self.v

    # Redefino la funcion que calcula las probabilidades condicionales.
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        v = self.V()
        token = tuple([token])
        if not prev_tokens:
            assert n == 1
            prev_tokens = []
        assert len(prev_tokens) == n - 1
        prev_tokens = tuple(prev_tokens)
        ngram = prev_tokens + token
        # Suavizado Add-One.
        ngram_prob = float(self.count(ngram) + 1)
        prev_tokens_prob = float(self.count(prev_tokens) + v)
        # No hay ZeroDivisionError
        conditional_prob = ngram_prob / prev_tokens_prob

        return conditional_prob
