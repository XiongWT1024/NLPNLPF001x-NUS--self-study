import numpy as np
from collections import defaultdict
import math

class MyMultinomialNaiveBayes:

    def __init__(self):
        # count is a dictionary which stores several dictionaries corresponding to each news category
        # each value in the subdictionary represents the freq of the key corresponding to that news category
        self.count = {}

        # classes represents the different news categories
        self.classes = None

        self.vocabulary = list()
        self.vocabulary_size = 0

    def build_vocabulary(self, texts, cutoff_freq):
        """
        Build vocabulary
        [Input] texts (list of list of string) e.g., [['loving', 'thing', 'good'], ......]
                cutoff_freq (int) only words with frequencies above the cutoff_freq were retained in the vocab. e.g., 2
        """
        vocabulary = []
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: You can use a dictionary object to record the frequency of each word.
        voc_dict_freq = defaultdict(int)
        for text in texts:
            for word in text:
                voc_dict_freq[word]+=1

        # Tips: For words with frequencies above the cutoff_freq, you can store them in the vocabulary object.
        vocabulary={word for word,freq in voc_dict_freq.items() if freq > cutoff_freq}


        ### Your code ends here #################################################################
        #########################################################################################
        self.vocabulary = vocabulary
        self.vocabulary_size = len(self.vocabulary)
        if vocabulary:
            print("Vocab size:", self.vocabulary_size)
        else:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")


    def texts2vec(self, texts):
        """
        Transform texts into vectors via self.vocabulary
        [input] texts (list of list of word) e.g., [['loving', 'thing', 'good'], ......]
        [return] vectorized_texts (list of list of int) e.g.,  [[1, 0, 0, ...], ......]
        """
        vectorized_texts = []
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: You may need a for loop here to process each `text` in `texts`
        for txt in texts:
            voc_vect = [0]*self.vocabulary_size
            for word in txt:
            # Tips: You might want to have a list of how many times each word appears.
                if word in self.vocabulary:
                    voc_vect[list(self.vocabulary).index(word)]+=1

            # Tips: For each word, find the corresponding position in the list and count it.
            vectorized_texts.append(voc_vect)


        ### Your code ends here #################################################################
        #########################################################################################
        if not vectorized_texts:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        return vectorized_texts


    def fit(self, vectorized_texts, labels):
        """
        Train Naive bayes classifier
        [input] vectorized_texts (list of list of int) e.g., [[1, 0, 0, ...], ......]
                labels (list of string)  e.g., ["approval", "joy", "joy", "sadness", ...]
        """
        self.count = {}
        self.classes = None
        
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tips: store all the results in the self.count and self.classes
        # Tip 1: classes represents the different emotion labels.
        self.classes = set(labels)
        

        # Tip 2: initialize self.count object, self.count can be a nested dictionary:
        # {
        #    "label 1": {
        #        0 : 0 # Used to count the number of occurrences of the first word in the vocabulary, correspond to count(w0, Nc)
        #        1 : 0 # Used to count the number of occurrences of the second word in the vocabulary, correspond to count(w0, Nc)
        #        ......
        #        total_words: 0 # correspond to count(w, Nc)
        #        total_datas: 0 # correspond to Nc
        #    },
        #    "label 1": {.....}
        #    "total_datas": 0 # correspond to Ndoc

        # }
        for label in self.classes:
            # Initialize counts for each label
            self.count[label] = defaultdict(int)
            self.count[label]['total_words'] = 0
            self.count[label]['total_data'] = 0

        for vector, label in zip(vectorized_texts, labels):
            # Update counts for each label based on the vector
            self.count[label]['total_data'] += 1
            for word_index, word_count in enumerate(vector):
                if word_index < self.vocabulary_size:
                    self.count[label][word_index] += word_count
                    self.count[label]['total_words'] += word_count
        # Add a count for total documents
        self.count['total_data'] = len(vectorized_texts)
        ### Your code ends here #################################################################
        #########################################################################################
        if not self.classes:
            print("Warning!!! You need to implement this function! This function accounts for 30 points!")
        else:
            print("Finish Model Training.")

    def probability(self, vectorized_text, one_class):
        """
        Given one label and one vectorized_text, calculate the log probability.
        [Input] vectorized_text (list of int) e.g., [1, 0, 0, ...]
                one_class (string) e.g., "joy"
        [Output] log_prob (float) log probability
        """
        log_prob = None
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tip 1: log P(c)
        total_docs = self.count['total_data']
        class_docs = self.count[one_class]['total_data']
        log_prob_class = math.log(class_docs / total_docs)
        log_prob_words = 0
        for word_index, word_count in enumerate(vectorized_text):
            word_frequency_in_class = self.count[one_class].get(word_index, 0)
            log_prob_words += word_count * math.log((word_frequency_in_class + 1) / (self.count[one_class]['total_words'] + self.vocabulary_size))
        log_prob = log_prob_class + log_prob_words
        # Tip 2: sum(log P(wi|c))

        ### Your code ends here #################################################################
        #########################################################################################
        if not log_prob:
            print("Warning!!! You need to implement this function! This function accounts for 20 points!")
        return log_prob

    def predict_single(self, vectorized_text):
        """
        Assign one label for a given vectorized text
        [input] vectorized_text (list of int) e.g., [1, 0, 0, ...]
        [output] prediction (string) e.g., "joy"
        """
        prediction = None
        #########################################################################################
        ### Your code starts here ###############################################################
        # Tip: for each label, calculate its log_prob and choose the best one.
        max_log_prob = float('-inf')
        for one_class in self.classes:
            log_prob = self.probability(vectorized_text, one_class)
            # Select the class with the highest log probability
            if log_prob > max_log_prob:
                max_log_prob = log_prob
                prediction = one_class



        ### Your code ends here #################################################################
        #########################################################################################
        if not prediction:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")
        return prediction

    def predict(self, vectorized_texts):
        """
        Assign labels for a given corpus
        [input] vectorized_texts (list of list of int) e.g., [[1, 0, 0, ...], ......]
        [output] predictions (list of string) e.g., ["joy", "approval", ......]
        """
        predictions = []
        for vectorized_text in vectorized_texts:
            prediction = self.predict_single(vectorized_text)
            predictions.append(prediction)
        return predictions

    def score(self, predictions, test_labels):
        """
        Accuracy calculation
        [Input] predictions (list of string) e.g., ["joy", "approval", ......]
                test_labels (list of string) e.g., ["joy", "approval", ......]
        [Output] acc (float) accuracy score
        """
        acc = None
        #########################################################################################
        ### Your code starts here ###############################################################
        correct_count = sum(predicted == actual for predicted, actual in zip(predictions, test_labels) if predicted == actual)
        acc = correct_count / len(test_labels)





        ### Your code ends here #################################################################
        #########################################################################################
        if not acc:
            print("Warning!!! You need to implement this function! This function accounts for 10 points!")
        return acc
