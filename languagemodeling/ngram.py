# https://docs.python.org/3/library/collections.html
from collections import defaultdict, OrderedDict
from math import log, floor
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
        self.v = 0  # Longitud del vocabulario.

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
        cond_prob = self.calculate_cond_prob(token, prev_tokens)

        # Todas las clases comparten el mismo codigo para la funcion cond_prob.
        # Basta con redefinir la funcion auxiliar basandose en como calcula
        # cada modelo las probabilidades condicionales entre dos eventos.

        return cond_prob

    def calculate_cond_prob(self, token, prev_tokens):
        """Auxiliar function to calculate conditional probabilities and to
        maximize code reusability.
        Theoretical base: Ngram Model.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
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
        # Obtengo la perplexity a partir de su cross_entropy
        perplexity = pow(2, self.cross_entropy(sents))

        return perplexity

    def cross_entropy(self, sents):
        """Aproximated Cross-entropy of a language model.

        sents -- abstraction of a 'long sentence'.
        """
        log_prob = self.log_probability(sents)
        # La entropia cruzada entre la distribucion de un modelo y la
        # distribucion de palabras en un corpus dado se puede aproximar
        # como la (-)log_prob de una secuencia larga normalizada por
        # el total de palabras del corpus.
        c_entropy = float(-1) / self.m * log_prob

        return c_entropy

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
        log_prob = self.sent_log_prob(long_sent)

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

    def V(self):
        """Size of the vocabulary.
        """
        return self.v

    def calculate_cond_prob(self, token, prev_tokens):
        """Auxiliar function to calculate conditional probabilities and to
        maximize code reusability.
        Theoretical base: Add-One Smoothing.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        # Redefino la funcion auxiliar que calcula la probabilidad condicional.
        ngram = prev_tokens + token
        # Suavizado Add-One.
        ngram_prob = float(self.count(ngram) + 1)
        prev_tokens_prob = float(self.count(prev_tokens) + self.V())
        # No hay ZeroDivisionError
        conditional_prob = ngram_prob / prev_tokens_prob

        return conditional_prob


class InterpolatedNGram(NGram):

    def __init__(self, n, sents, gamma=None, addone=True):
        """
        n -- order of the model.
        sents -- list of sentences, each one being a list of tokens.
        gamma -- interpolation hyper-parameter (if not given, estimate using
            held-out data).
        addone -- whether to use addone smoothing (default: True).
        """
        assert n > 0
        self.n = n
        self.counts = counts = defaultdict(int)
        self.v = 0
        self.addone = addone
        self.gamma = gamma

        # Delimitadores
        for sent in sents:
            self._delimiters(sent, n)

        # Held-out data para calcular el valor de gamma optimo.
        if not self.gamma:
            # Sumo 1 a index para el caso en que len(sents) < 10.
            held_out_index = int(floor(len(sents) * 10 / 100) + 1)
            held_out_data = sents[-held_out_index:]  # Las ultimas 10% sents.
            sents = sents[:-held_out_index]  # Primer 90%.

        if self.addone:
            # Calcula la longitud del vocabulario.
            _vocabulary = set()
            for sent in sents:
                    for token in sent:
                        if token != BEGIN:
                            _vocabulary.add(token)
            self.v = len(_vocabulary)

        # Incializo los counts. (Si no hay gamma las sentencias son held_out).
        for sent in sents:
            # Creo el diccionario de frecuencias para
            # ([n],[(n-1)],...,[(n-i)],...,[1])-gramas.
            for i in range(len(sent) - n + 1):
                ngram = tuple(sent[i: i + n])
                # (n-j)-gramas con j = 0,...,n.
                for j in range(n + 1):
                    counts[ngram[:n - j]] += 1

            # Para el caso n > 1 agrego los counts de los delimitadores de fin.
            if self.n > 1:
                counts[tuple([END, ])] += 1

        # Calculo el gamma optimo si no viene como argumento.
        if not self.gamma:
            self.maximize_gamma(held_out_data)

    def count(self, tokens):
        """Count for an n-gram or (n-1)-gram.

        tokens -- the n-gram,(n-1)-gram,...,1-gram tuple. (() Included).
        """
        # Redefino la funcion count para evitar borrar el assert de la clase
        # NGram que controla tuplas de longitudes exclusivas n y (n-1).
        assert len(tokens) in range(self.n + 1)
        return self.counts[tokens]

    def maximize_gamma(self, sents):

        # Rango de valores 'hardcodeados a ojo'.
        # Se podria definir una funcion que incremente de manera variable los
        # valores de gamma, aumentando el delta de incremento hasta encontrar
        # 'picos' que indiquen que la funcion deja de crecer. Luego moverse en
        # dichos intervalos siguiendo la idea de busqueda binaria hasta hallar
        # un valor maximo.
        # A fines practicos se hardcodean los valores en base a los resultados
        # de las pruebas a mano con distintos valores para los corpus usados.
        gammas_list = [0, 1, 9, 200, 500, 1000, 2000, 5000, 7500, 9000, 10.000]
        logs_prob_list = []

        # Calculo log_prob para cada gamma.
        for gamma in gammas_list:
            self.gamma = gamma
            # Cada nuevo gamma influye de manera directa en la log_probability.
            logs_prob_list.append(self.log_probability(sents))

        # Elijo como gamma aquel que la maximice.
        self.gamma = gammas_list[logs_prob_list.index(max(logs_prob_list))]

    def lambdas_list(self, ngram):
        """Lambda values for an ngram.

        ngram -- (prev_tokens + token) tuple.
        """
        n = self.n
        # Formula teorica para calcular lambdas:
        # Lambda_i = (1 - sum (Lamda_j)) * C(Xi...Xn-1)/(C(Xi...Xn-1) + Gamma).
        lambdas = [1]  # Lista de valores. Lambda_0 es 1.
        for i in range(1, n + 1):
            lambda_i = lambdas[0]
            for j in range(1, i):  # Incluye el valor (i-1).
                # Sumatoria de lambdas_j.
                lambda_i -= lambdas[j]
            # Factor comun para cada lambda_i entre 1 y n-1.
            if i < n:
                numerator_i = self.count(ngram[i - 1: -1])  # Xi,...,Xn-1.
                denominator_i = float(numerator_i + self.gamma)

                lambda_i *= numerator_i / denominator_i
            # Agrego a la lista el nuevo valor calculado.
            lambdas.append(lambda_i)

        return lambdas

    def q_mls_list(self, ngram):
        """Maximum-likelihood values for an ngram.

        ngram -- (prev_tokens + token) tuple.
        """
        q_mls = [0]  # Para anular en la sumatoria (***) el valor de lambda_0.
        for i in range(1, self.n + 1):
            # Addone para el nivel mas bajo. (Unigramas).
            add_one = self.addone and len(ngram[i - 1:])  # Suma 0 solo si and.
            add_v = add_one * self.v  # Si add_one es 1 suma V, c/contrario 0.
            numerator_i = self.count(ngram[i - 1:])  # Xi,...,Xn-1,Xn.
            denominator_i = float(self.count(ngram[i - 1: -1]))  # Xi,...,Xn-1.

            q_mls.append((numerator_i + add_one) / (denominator_i + add_v))

        return q_mls

    def calculate_cond_prob(self, token, prev_tokens):
        """Auxiliar function to calculate conditional probabilities and to
        maximize code reusability.
        Theoretical base: Interpolation Smoothing.

        token -- the token.
        prev_tokens -- the previous n-1 tokens (optional only if n = 1).
        """
        ngram = prev_tokens + token  # Genero ngrama y aplico interpolacion.
        try:
            # Calculo los valores de los n lambdas y qMLs.
            lambdas_list = self.lambdas_list(ngram)
            q_mls_list = self.q_mls_list(ngram)

            # Probabilidad condicional para Interpolacion.
            assert len(lambdas_list) == len(q_mls_list)
            # Sumatoria (***) de lambas_i y qML para cada cond_prob.
            cond_prob = sum([a * b for a, b in zip(lambdas_list, q_mls_list)])

        except ZeroDivisionError:
            cond_prob = 0

        return cond_prob
