import warnings
warnings.filterwarnings('ignore',
                        message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class CorpusAnalyser:
    def __init__(self, corpus: pd.DataFrame, column: str):
        """
        Initialisiert die CorpusAnalyser Klasse, berechnet die TF-IDF-Matrix für den vorverarbeiteten Korpus
        und erstellt einen invertierten Index für die Token im Korpus.

        :param corpus: Ein DataFrame, der die vorverarbeitete Dokumentensammlung enthält (jedes Dokument ist eine Liste von Tokens).
        :param column: Der Spaltenname im DataFrame, der die vorverarbeiteten Dokumente enthält.
        """
        self.corpus = corpus
        self.column = column
        self.inverted_index = self.create_inverted_index()

        # Initialisierung des TfidfVectorizers
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus[column])
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

    def create_inverted_index(self):
        """
        Erstellt einen invertierten Index für die Token im Korpus.

        :return: Ein Dictionary, das jedes Token seinem Vorkommen in Dokumenten zuordnet.
        """
        inverted_index = {}
        for i, tokens in enumerate(self.corpus[self.column]):
            for token in tokens:
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append(i)
        return inverted_index

    def corpus_number_of_texts(self):
        """
        Zählt die Anzahl der Texte im Korpus.

        :return: Die Anzahl der Texte im Korpus.
        """
        return len(self.corpus)

    def corpus_average_text_length(self):
        """
        Berechnet die durchschnittliche Textlänge (in Tokens) im Korpus.

        :return: Die durchschnittliche Textlänge im Korpus.
        """
        total_tokens = sum(len(text) for text in self.corpus[self.column])
        return total_tokens / len(self.corpus)

    def get_n_highest_tfidf_ids(self, term: str, n: int):
        """
        Gibt die Indizes der Dokumente mit den n höchsten TF-IDF-Werten für einen bestimmten Term zurück.

        :param term: Das Wort, für das die TF-IDF-Werte abgerufen werden sollen.
        :param n: Die Anzahl der Dokumente, die zurückgegeben werden sollen.
        :return: Ein DataFrame mit den Indizes und TF-IDF-Werten der Top-n Dokumente für den Term.
        """
        if term in self.feature_names:
            term_index = list(self.feature_names).index(term)
            # Extrahieren der TF-IDF-Werte für den Term und konvertieren in ein DataFrame
            term_tfidf_values = pd.DataFrame(self.tfidf_matrix[:, term_index].toarray(), columns=[term])
            # Hinzufügen der Dokumentenindizes zum DataFrame
            term_tfidf_values['doc_id'] = term_tfidf_values.index
            # Sortieren der Werte und Auswahl der Top-n Einträge
            top_n = term_tfidf_values.sort_values(by=term, ascending=False).head(n)
            return top_n[['doc_id', term]]
        else:
            print(f"'{term}' nicht im Vokabular gefunden.")
            return pd.DataFrame()

    def get_tfidf_matrix(self):
        """
        Visualisiert die TF-IDF-Matrix als DataFrame.

        :return: Die TF-IDF-Matrix als DataFrame.
        """
        return pd.DataFrame(self.tfidf_matrix.toarray(), columns=self.feature_names)

    def find_similar_documents(self, doc_id: int, n: int):
        """
        Findet die n ähnlichsten Dokumente zu einem bestimmten Dokument anhand der Kosinus-Ähnlichkeit.

        :param doc_id: Die ID des Dokuments, für das ähnliche Dokumente gefunden werden sollen.
        :param n: Die Anzahl der ähnlichen Dokumente, die zurückgegeben werden sollen.
        :return: Ein DataFrame mit den Indizes und Kosinus-Ähnlichkeiten der ähnlichsten Dokumente.
        """
        # Berechnung der Kosinus-Ähnlichkeiten
        cosine_similarities = self.tfidf_matrix.dot(self.tfidf_matrix[doc_id].T).toarray().ravel()
        # Sortieren der Ähnlichkeiten und Auswahl der Top-n Einträge
        similar_docs = pd.DataFrame(
            {'doc_id': list(range(len(cosine_similarities))), 'cosine_similarity': cosine_similarities})
        similar_docs = similar_docs.sort_values(by='cosine_similarity', ascending=False).head(n)
        return similar_docs

    def search_word(self, substring):
        """
        Sucht nach einem Substring (Token) in den Texten des Korpus und gibt die Dokumentenindizes und die Positionen zurück.

        :param substring: Der zu suchende Substring (Token).
        :return: Ein Dictionary, das jeden Dokumentenindex enthält, in dem der Substring (Token) vorkommt, zusammen mit den Positionen des Substrings (Tokens) in diesem Dokument.
        """
        # using the inverted index
        return self.inverted_index




if __name__ == '__main__':
    # Beispiel-Korpus
    corpus = pd.read_csv('../data/train/processed_train.csv')
    # Initialisierung des CorpusAnalyser
    analyser = CorpusAnalyser(corpus, 'comment_text')
    # Anzahl der Texte im Korpus
    print("Anzahl Texte in Korpus:", analyser.corpus_number_of_texts())
    # Durchschnittliche Textlänge
    print("Durchschnittliche Anzahl Wörter in den Texten:", analyser.corpus_average_text_length())
    # TF-IDF-Werte für ein bestimmtes Wort
    print(analyser.get_n_highest_tfidf_ids('vandalism', 5))
    #print(analyser.search_word("vandalism"))
