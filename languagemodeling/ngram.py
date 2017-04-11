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
            # Agrego n-1 delimitadores de inicio y uno de fin de sentencia
            sent[0:0] = (n-1) * ['<s>']
            sent.append('</s>')
            # Creo el diccionario de n-gramas y (n-1)-gramas y sus frecuencias.
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                # Incremento la frecuencia del n-grama.
                counts[ngram] += 1
                # Con n = 1 cargaria la tupla vacia con N (Numero total de tokens).
                if n != 1:
                    # Incrementa frecuencia del (n-1)-grama.
                    counts[ngram[:-1]] += 1

#    def prob(self, token, prev_tokens=None):
#        n = self.n
#        if not prev_tokens:
#            prev_tokens = []
#        assert len(prev_tokens) == n - 1
#
#        tokens = prev_tokens + [token]
#        return float(self.counts[tuple(tokens)]) / self.counts[tuple(prev_tokens)]

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.
 
        tokens -- the n-gram or (n-1)-gram tuple.
        """
        n = self.n
        # La tupla nunca puede ser vacía.
        assert len(tokens) != 0
        # Chequeo n-uplas o (n-1)-uplas.
        assert len(tokens) == n or len(tokens) == n - 1
        # Frecuencia asociada a la tupla que pasa como argumento.
        return self.counts[tokens]

 
    def cond_prob(self, token, prev_tokens=None):
        """Conditional probability of a token.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        n = self.n
        # El primer argumento es un unico token.
        ##assert len(token) == 1
        # Numero de apariciones del token.
        token_prob = self.count(token)
        # Caso unigramas.
        if n == 1:

            return token_prob
        # Casos n > 1.
        conditional_prob = 0
        # Chequeo largo de tupla de tokens previos.
        assert len(prev_tokens) == n - 1
        # Tupla que forma el n-grama.
        # Probabilidades.
        ngram_prob = float(self.count(ngram)) # P(Interseccion de los eventos).
        prev_tokens_prob = float(self.count(prev_tokens)) # P(Eventos condicionantes).
        # Regla de la probabilid condicional.
        conditional_prob =  ngram_prob / prev_tokens_prob 

        return conditional_prob


        # TENER EN CUENTA EL CASO PROB = 0???
        # OCURRE REALMENTE ALGUNA VEZ.???
        #if (prev_tokens_prob == 0):
        #try: zerodivisionerror

            #return conditional_proB


    def sent_prob(self, sent):
        """Probability of a sentence. Warning: subject to underflow problems.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        # Delimitadores de inicio y fin.
        sent[0:0] = (n-1) * ['<s>']
        sent.append('</s>')

        prob = 1
        # Recorro cada n-grama en busca de sus probabilidades.
        for i in range(len(sent) - n + 1):
            # Formo las tuplas argumentos.
            prev_tokens = tuple(sent[i: i + n - 1]) # (n-1)grama de tk previos.
            token = tuple(sent[i+n]) # Token i.
            # Markov Assumption
            prob *= self.cond_prob(token, prev_tokens)
        
        return prob


    def sent_log_prob(self, sent):
        """Log-probability of a sentence.
 
        sent -- the sentence as a list of tokens.
        """
        n = self.n
        # Delimitadores de inicio y fin.
        sent[0:0] = (n-1) * ['<s>']
        sent.append('</s>')

        prob = 0
        for i in range(len(sent) - n + 1):
            prev_tokens = tuple(sent[i: i + n - 1])
            token = tuple(sent[i+n])
            # Propiedad del logaritmo + Markov Assumption 
            try:
                # Log base 2
                prob += log(self.cond_prob(token, prev_tokens),2)
            # Cuando prob es 0.
            except ZeroDivisionError:
                prob -= float('inf')

