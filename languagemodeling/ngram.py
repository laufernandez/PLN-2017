# https://docs.python.org/3/library/collections.html
from collections import defaultdict
from math import log


class NGram(object):

    def __init__(self, n, sents):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)

        for sent in sents:
            # Delimitadores de inicio y fin de sentencia.
            self.delimiters(sent, n)
            # Creo el diccionario de n-gramas y (n-1)-gramas y sus frecuencias.
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                # Incremento la frecuencia del n-grama.
                counts[ngram] += 1
                # Incrementa frecuencia del (n-1)-grama.
                counts[ngram[:-1]] += 1

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
        self.delimiters(sent, n)

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
        self.delimiters(sent, n)

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

    def delimiters(self, sent, n):
        """ Add delimiters to a sentence.

        <s> -- beg delimiters.
        </s> -- end delimiters.
        """
        # Agrego n-1 delimitadores de inicio y uno de fin de sentencia
        sent[0:0] = (n-1) * ['<s>']
        sent.append('</s>')


class NGramGenerator(object):

    def __init__(self, model):
        """
        model -- n-gram model.
        """

    def generate_sent(self):
        """Randomly generate a sentence."""

    def generate_token(self, prev_tokens=None):
        """Randomly generate a token, given prev_tokens.

        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
