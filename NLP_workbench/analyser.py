from collections import Counter


class TextAnalyser:
    def __init__(self):
        pass

    def word_count(self, text):
        """
        Zählt die Wörter im Text.

        :return: Die Anzahl der Wörter im Text.
        """
        return len(text)

    def search_substring(self, text, substring):
        """
        Sucht nach einem Substring in den Wörtern des Textes und gibt die Indizes zurück.

        :param substring: Der zu suchende Substring.
        :return: Eine Liste von Indizes, an denen der Substring gefunden wurde.
        """
        return [i for i, word in enumerate(text) if substring in word]

    def most_common_words(self, text, n):
        """
        Bestimmt die n häufigsten Wörter im Text.

        :param n: Die Anzahl der zurückzugebenden häufigsten Wörter.
        :return: Eine Liste der n häufigsten Wörter und ihre Häufigkeiten.
        """
        word_counts = Counter(text)
        return word_counts.most_common(n)

    def word_frequency(self, text, word):
        """
        Berechnet die Frequenz eines bestimmten Wortes im Text.

        :param word: Das Wort, dessen Frequenz berechnet werden soll.
        :return: Die Frequenz des Wortes im Text.
        """
        return text.count(word) / len(text)

    def average_word_length(self, text):
        """
        Berechnet die durchschnittliche Wortlänge im Text.

        :return: Die durchschnittliche Wortlänge.
        """
        word_lengths = [len(word) for word in text]
        return sum(word_lengths) / len(word_lengths)

    def sentence_count(self, text):
        """
        Zählt die Sätze im Text, basierend auf Punktuation (., !, ?).

        :return: Die Anzahl der Sätze im Text.
        """
        return sum(text.count(punct) for punct in '.!?')


if __name__ == '__main__':
    text = ["This", "is", "a", "test", "text", ".", "It", "is", "a", "test", "text", "."]
    ta = TextAnalyser()
    print(ta.word_count(text))
    print(ta.search_substring(text, "is"))
    print(ta.most_common_words(text, 2))
    print(ta.word_frequency(text, "is"))
    print(ta.average_word_length(text))
    print(ta.sentence_count(text))
