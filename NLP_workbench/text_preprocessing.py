import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('words')

from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


class TextPreprocessing:
    """
    Diese Klasse dient der Vorverarbeitung von Texten. Sie bietet Methoden zur Tokenisierung, Entfernung von Sonderzeichen,
    Entfernung von Stopwords, Stemming und Lemmatisierung.

    Attributes:
        dictionary: Liste mit Wörtern
        stops: Liste mit Stopwords
        stemmer: Stemmer
        lemmatizer: Lemmatizer
    """

    def __init__(self):
        self.stops = set(stopwords.words('english'))
        self.dictionary = set(words.words())
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

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

    def remove_numbers(self, text):
        """
        Diese Methode entfernt Zahlen aus dem Text.
        :param text:
        :return:
        """
        return re.sub(r'\d+', '', text)

    def lemmatization(self, text: list):
        """
        Diese Methode führt die Lemmatisierung auf den Text aus, wobei die Wörter auf ihre Grundform reduziert werden.
        :param text:
        :return:
        """
        return [self.lemmatizer.lemmatize(word) for word in text]

    def stemming(self, text: list):
        """
        Diese Methode führt das Stemming auf den Text aus, wobei die Wörter auf ihren Wortstamm reduziert werden. Ist
        per Default deaktiviert, da Lemmatisierung in der Regel bessere Ergebnisse liefert. Stemming kann jedoch
        in manchen Fällen sinnvoll sein, gerade in Performance-sensitiven Anwendungen.
        :param text:
        :return:
        """
        return [self.stemmer.stem(word) for word in text]

    def check_if_valid_word(self, text: list):
        """
        Diese Methode prüft, ob ein Wort ein gültiges Wort ist. Achtung Fluchwörter werden gefiltert.
        Verwendet wird die Liste der Wörter aus dem NLTK-Korpus.
        :param text:
        :return:
        """
        return [word for word in text if word in self.dictionary]

    def preprocess_text(self, text, lower_case=True, remove_special_chars=True, remove_numbers=True,
                        remove_punctuation=True,
                        remove_stopwords=True, lemmatization=True, stemming=False, check_valid_word=False):
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
        if remove_numbers:
            text = self.remove_numbers(text)
        if remove_punctuation:
            text = self.remove_punctuation(text)
        if remove_stopwords:
            text = self.remove_stopwords(text)
        if lemmatization:
            text = self.lemmatization(text)
        if stemming:
            text = self.stemming(text)
        if check_valid_word:
            text = self.check_if_valid_word(text)

        return text


if __name__ == "__main__":
    text = 'This error will persist for a long time as it continues to reproduce... The latest reproduction I know is from ENCYCLOPEDIA BRITANNICA ALMANAC 2008 wich states '
    test = TextPreprocessing()
    print(test.preprocess_text(text))
