import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessing:
    """
    Diese Klasse dient der Vorverarbeitung von Texten. Sie bietet Methoden zur Tokenisierung, Entfernung von Sonderzeichen,
    Entfernung von Stopwords, Stemming und Lemmatisierung.

    Attributes:
        stops: Liste mit Stopwords
        stemmer: Stemmer
        lemmatizer: Lemmatizer
    """

    def __init__(self):
        self.stops = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def lower_text(self, text):
        """
        Diese Methode wandelt den Text in Kleinbuchstaben um.
        :param text:
        :return:
        """
        return text.lower()

    def tokenize_text(self, text):
        """
        Diese Methode tokenisiert den Text. Dazu wird der Text in Wörter zerlegt und in Kleinbuchstaben umgewandelt.
        :param text:
        :return:
        """
        return word_tokenize(text)

    def remove_special_chars(self, text):
        """
        Diese Methode entfernt Sonderzeichen aus dem Text.
        :param text:
        :return:
        """
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def remove_punctuation(self, text):
        """
        Diese Methode entfernt Satzzeichen aus dem Text.
        :param text:
        :return:
        """
        return re.sub(r'[^\w\s]', '', text)

    def remove_stopwords(self, text):
        """
        Diese Methode entfernt Stopwords aus dem Text. Die Stopwords kommen aus dem NLTK-Korpus.
        :param text:
        :return:
        """
        text = self.tokenize_text(text)
        return [word for word in text if word not in self.stops]

    def lemmatization(self, text: list):
        """
        Diese Methode führt die Lemmatisierung auf den Text aus, wobei die Wörter auf ihre Grundform reduziert werden.
        :param text:
        :return:
        """
        return [self.lemmatizer.lemmatize(word) for word in text]

    def preprocess_text(self, text, lower_case=True, remove_special_chars=True, remove_punctuation=True,
                        remove_stopwords=True, lemmatization=False):
        """
        Diese Methode führt die Vorverarbeitung auf den Text aus.
        :param text:
        :param remove_special_chars:
        :param remove_punctuation:
        :param remove_stopwords:
        :param lemmatization:
        :return:
        """
        if lower_case:
            text = self.lower_text(text)
        if remove_special_chars:
            text = self.remove_special_chars(text)
        if remove_punctuation:
            text = self.remove_punctuation(text)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        if lemmatization:
            text = self.lemmatization(text)
        return text


if __name__ == "__main__":
    text = 'This error will persist for a long time as it continues to reproduce... The latest reproduction I know is from ENCYCLOPÃ†DIA BRITANNICA ALMANAC 2008 wich states '
    test = TextPreprocessing()
    print(test.preprocess_text(text))
