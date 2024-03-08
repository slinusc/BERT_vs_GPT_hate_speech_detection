from collections import Counter
from textblob import TextBlob


class TextAnalyser:
    def __init__(self):
        pass

    @staticmethod
    def word_count(text):
        """
        Zählt die Wörter im Text.

        :return: Die Anzahl der Wörter im Text.
        """
        return len(text)

    @staticmethod
    def search_substring(text, substring):
        """
        Sucht nach einem Substring in den Wörtern des Textes und gibt die Indizes zurück.

        :param substring: Der zu suchende Substring.
        :return: Eine Liste von Indizes, an denen der Substring gefunden wurde.
        """
        return [i for i, word in enumerate(text) if substring in word]

    @staticmethod
    def most_common_words(text, n):
        """
        Bestimmt die n häufigsten Wörter im Text.

        :param n: Die Anzahl der zurückzugebenden häufigsten Wörter.
        :return: Eine Liste der n häufigsten Wörter und ihre Häufigkeiten.
        """
        word_counts = Counter(text)
        return word_counts.most_common(n)

    @staticmethod
    def word_frequency(text, word):
        """
        Berechnet die Frequenz eines bestimmten Wortes im Text.

        :param word: Das Wort, dessen Frequenz berechnet werden soll.
        :return: Die Frequenz des Wortes im Text.
        """
        return text.count(word) / len(text)

    @staticmethod
    def average_word_length(text):
        """
        Berechnet die durchschnittliche Wortlänge im Text.

        :return: Die durchschnittliche Wortlänge.
        """
        word_lengths = [len(word) for word in text]
        return sum(word_lengths) / len(word_lengths)

    @staticmethod
    def sentence_count(text):
        """
        Zählt die Sätze im Text, basierend auf Punktuation (., !, ?).

        :return: Die Anzahl der Sätze im Text.
        """
        return sum(text.count(punct) for punct in '.!?')

    @staticmethod
    def sentiment_analysis(text: str):
        """
        Führt eine Sentimentanalyse des gegebenen Textes durch und gibt die Polarität und Subjektivität zurück.

        :param text: Der Text, der analysiert werden soll.
        :return: Ein Dictionary mit Polarität und Subjektivität des Textes.
        """
        if isinstance(text, list):
            text = ' '.join(text)

        analysis = TextBlob(text)
        sentiment = analysis.sentiment
        return {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}


if __name__ == '__main__':
    text = ["This", "is", "a", "good", "test", "text", ".", "It", "is", "a", "test", "text", "."]
    ta = TextAnalyser()
    print(ta.word_count(text))
    print(ta.search_substring(text, "is"))
    print(ta.most_common_words(text, 2))
    print(ta.word_frequency(text, "is"))
    print(ta.average_word_length(text))
    print(ta.sentence_count(text))
    print(ta.sentiment_analysis(text))
