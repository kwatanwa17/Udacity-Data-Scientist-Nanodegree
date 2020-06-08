# use StartingVerbExtractor
import pandas as pd

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    """
    This function tokenize a text.

    :param text: a text to tokenize
    :return: a list of tokens
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """
        This function return 1 if a text starts with a verb

        :param text: a text to extract
        :return: 1 if the text starts with a verb, 0 otherwise
        """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        """
        This function fits data to its targets

        :param x: data
        :param y: targets
        """
        return self

    def transform(self, X):
        """
        This function transforms data to dataframe

        :param X: a array or list to transform
        :return: a transformed dataframe
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
