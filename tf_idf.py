import math
import numpy as np
import pandas as pd

from collections import Counter


class TFIDF:
    """Calculate the term frequency  - inverse data frequency (TF-IDF) for a given corpus."""

    def __init__(self, corpus: pd.DataFrame):
        # The original Data Frame
        self.corpus = corpus

        # A dataframe to hold the tf-idf values
        self.df_tfidf = pd.DataFrame()

        # A set of all the words in the corpus
        self.words = set()

        # create initial word list
        self.generate_wordlist()

    def generate_wordlist(self):
        # Assume data is in column 1
        # Assumes each cell in column 1 is a string

        # Loop through rows of dataframe
        # df.values returns the columns as 2D array
        # because we have only one column, squeeze flattens the matrix
        # so we can loop through the column
        # and document becomes the actual string
        for document in self.corpus.values.squeeze():
            # Split string and add to words set
            self.words.update(document.split())

    def calculate_tf(self):
        # Create a Dataframe containing the Term Frequency values

        # Loop through rows of dataframe by index i.e. from 0 to number of rows
        for i in range(0, self.corpus.shape[0]):
            # Get row contents as series using iloc[] and index position of row
            row = self.corpus.iloc[i]

            # The number of words in the string
            number_words = len(row[0].split())

            # The count of each word
            words_count = Counter(row[0].split())

            # Create a zero value dictionary of all words in words list
            word_dict = dict.fromkeys(self.words, 0)

            # Update the word dictionary with the count of word / number of words
            for word, count in words_count.items():
                word_dict[word] = count / number_words

            # Append the word dictionary to the df_tfidf Dataframe
            self.df_tfidf = self.df_tfidf.append(word_dict, ignore_index=True)

    def calculate_tfidf(self):
        # Start by calculating the Term Frequency Dataframe
        self.calculate_tf()

        # IDF = Log[ (Number of documents) /
        #            (Number of documents containing the word) ]
        number_documents = self.df_tfidf.shape[0]

        # Iterate over all the columns
        for column_name in self.df_tfidf:

            # How many rows in the column don't contain zero
            documents_with_words = (
                number_documents
                - self.df_tfidf[self.df_tfidf[column_name] == 0].shape[0]
            )

            word_tfidf = math.log10(number_documents / documents_with_words)

            # Update df_tfidf Dataframe with idf calculation
            self.df_tfidf[column_name] = self.df_tfidf[column_name] * word_tfidf


df = pd.read_csv("samples_02.txt", header=None)

tf_idf = TFIDF(df)
tf_idf.calculate_tfidf()
print(tf_idf.corpus)
print(tf_idf.words)
print(tf_idf.df_tfidf)
