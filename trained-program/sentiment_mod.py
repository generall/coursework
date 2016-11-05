import pickle
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from createuci import get_config
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from abc import *

config = get_config()
algos_folder = config['tone_analyse']['folder_for_algos']
negative_train_file = config['tone_analyse']['negative_train_file']
positive_train_file = config['tone_analyse']['positive_train_file']


def get_file_data(filepath, open_mode):
    try:
        file = open(filepath, open_mode)
        file_data = pickle.load(file)
        file.close()
        return file_data
    except (FileNotFoundError, EOFError):
        f = open(filepath, "w")
        f.close()
        return None


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# функция сохранения в файл
def save_to_file(filepath, open_mode, data):
    save_file = open(filepath, open_mode)
    pickle.dump(data, save_file)
    save_file.close()
    return


word_features = get_file_data(algos_folder + "word_features5k.pickle", "rb")


class SemanticAnaliser(metaclass=ABCMeta):
    def __init__(self):
        self.classifier = None
        self.name = None
        self.filename = None
        self.trained_classifier = None

    @abstractmethod
    def analise_text(self, text):
        pass

    @abstractmethod
    def learn(self, array_of_texts):
        pass

    def train_and_save_classifier(self, training_set, testing_set):
        self.classifier.train(training_set)
        print("%s accuracy percent:" % self.name,
              (nltk.classify.accuracy(self.classifier, testing_set)) * 100)
        save_to_file(algos_folder + self.filename, "wb", self.classifier)
        return


# наивный байесовский классификатор
class OriginalNBAnaliser(SemanticAnaliser):
    def __init__(self):
        self.classifier = nltk.NaiveBayesClassifier
        self.filename = "originalnaivebayes5k.pickle"
        self.name = "Original Naive Bayes"
        self.trained_classifier = get_file_data(
            algos_folder + self.filename, "rb")

    def analise_text(self, text):
        feats = find_features(text)
        return self.trained_classifier.classify(feats)

    def learn(self, training_set, testing_set):
        self.classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("%s accuracy percent:" % self.name,
              (nltk.classify.accuracy(self.classifier, testing_set)) * 100)
        self.classifier.show_most_informative_features(15)
        save_to_file(algos_folder + self.filename, "wb", self.classifier)


# Multinomial наивный байесовский классификатор
class MNBAnaliser(SemanticAnaliser):
    def __init__(self):
        self.classifier = SklearnClassifier(MultinomialNB())
        self.filename = "MNB_classifier5k.pickle"
        self.name = "MNB_classifier"
        self.trained_classifier = get_file_data(
            algos_folder + self.filename, "rb")

    def analise_text(self, text):
        feats = find_features(text)
        return self.trained_classifier.classify(feats)

    def learn(self, training_set, testing_set):
        self.train_and_save_classifier(training_set, testing_set)


# Bernoulli наивный байесовский классификатор
class BernoulliNBAnaliser(SemanticAnaliser):
    def __init__(self):
        self.classifier = SklearnClassifier(BernoulliNB())
        self.filename = "BernoulliNB_classifier5k.pickle"
        self.name = "BernoulliNB_classifier"
        self.trained_classifier = get_file_data(
            algos_folder + self.filename, "rb")

    def analise_text(self, text):
        feats = find_features(text)
        return self.trained_classifier.classify(feats)

    def learn(self, training_set, testing_set):
        self.train_and_save_classifier(training_set, testing_set)


# классификатор логической регрессии
class LogisticRegressionAnaliser(SemanticAnaliser):
    def __init__(self):
        self.classifier = SklearnClassifier(LogisticRegression())
        self.filename = "LogisticRegression_classifier5k.pickle"
        self.name = "LogisticRegression_classifier"
        self.trained_classifier = get_file_data(
            algos_folder + self.filename, "rb")

    def analise_text(self, text):
        feats = find_features(text)
        return self.trained_classifier.classify(feats)

    def learn(self, training_set, testing_set):
        self.train_and_save_classifier(training_set, testing_set)


# Linear Support Vector классификатор
class LinearSVCAnaliser(SemanticAnaliser):
    def __init__(self):
        self.classifier = SklearnClassifier(LinearSVC())
        self.filename = "LinearSVC_classifier5k.pickle"
        self.name = "LinearSVC_classifier"
        self.trained_classifier = get_file_data(
            algos_folder + self.filename, "rb")

    def analise_text(self, text):
        feats = find_features(text)
        return self.trained_classifier.classify(feats)

    def learn(self, training_set, testing_set):
        self.train_and_save_classifier(training_set, testing_set)


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


class MajorityVotingUnit:
    def __init__(self):
        self.analisers = []

    def add_analiser(self, analiser):
        self.analisers.append(analiser)

    def classify(self, text):
        votes = []
        for analiser in self.analisers:
            votes.append(analiser.analise_text(text))
        return mode(votes)

    def learn_all(self, training_set, testing_set):
        for analiser in self.analisers:
            analiser.learn(training_set, testing_set)


class VotingUnitWithProbability(MajorityVotingUnit):
    def __init__(self):
        super().__init__()

    def classify(self, text):
        votes = []
        for analiser in self.analisers:
            votes.append(analiser.analise_text(text))
        choice_votes = votes.count(mode(votes))
        probability = choice_votes / len(votes)
        return mode(votes), probability


NBclassifier = OriginalNBAnaliser()
MNB_classifier = MNBAnaliser()
BernoulliNB_classifier = BernoulliNBAnaliser()
LogisticRegression_classifier = LogisticRegressionAnaliser()
LinearSVC_classifier = LinearSVCAnaliser()
MainClassifier = VotingUnitWithProbability()
MainClassifier.add_analiser(NBclassifier)
MainClassifier.add_analiser(MNB_classifier)
MainClassifier.add_analiser(BernoulliNB_classifier)
MainClassifier.add_analiser(LogisticRegression_classifier)
MainClassifier.add_analiser(LinearSVC_classifier)


# Функция тонального анализа
def sentiment(text):
    return MainClassifier.classify(text)
