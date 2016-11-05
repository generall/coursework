from gensim import corpora
from nltk.corpus import stopwords
from collections import defaultdict
import re
from nltk.stem.wordnet import WordNetLemmatizer
import wikipedia
import requests
import os
import random
import math
import json
import argparse


def get_config():
    with open("config.json", "r") as json_data_file:
        config = json.load(json_data_file)
    return config


config = get_config()
categories = config['topic_modeling']['categories']
corpus_name = config['topic_modeling']['corpus_name']

stop = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()


class Text:
    def __init__(self):
        pass

    def is_word_correct(self, word):
        if not word.isdigit():
            if word not in stop:
                if len(word) > 2:
                    return True
        return False

    def capitalize_first_letter(self, string):
        if string:
            return string[0].upper() + string[1:]
        return string[:]

    def text_cleaning(self, text_for_clean):
        start = text_for_clean.rfind('http')
        if start != -1:
            end = text_for_clean.rfind('\n')
            if end != -1:
                text_for_clean = text_for_clean[:start] + text_for_clean[end:]
        text = re.sub(r'\([^\)]+\)', '', text_for_clean)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lstrip()
        text = text.rstrip()
        return text.lower()


class WikiText(Text):
    def __init__(self, list_of_categories, max_files_count_for_saving=45):
        super().__init__()
        self.max_files_count_for_saving = max_files_count_for_saving
        self.list_of_categories = list_of_categories

    def get_textfiles_from_wiki(self):
        for category in self.list_of_categories:
            parameters = {'action': 'query', 'list': 'categorymembers',
                          'cmtitle': 'Category:' + category, 'format': 'json',
                          'cmlimit': self.max_files_count_for_saving}
            r = requests.get(
                'https://en.wikipedia.org/w/api.php', params=parameters)
            returned_data = r.json()['query']['categorymembers']
            themes = []
            for article in returned_data:
                if int(article['ns']) == 0:
                    themes.append([article['title'], article['pageid']])
            if not os.path.exists(category):
                os.mkdir(category)
            for theme, pageid in themes:
                try:
                    page = wikipedia.page(pageid=pageid)
                    with open('./' + category + '/' + theme + '.wiki.txt', 'w') as file:
                        file.write(page.content)
                except AttributeError:
                    pass
        return

    def save_test_files(self, files_for_test, category):
        documents_for_test = []
        for file_name in files_for_test:
            with open('./' + category + '/' + file_name, 'r') as file:
                for line in file.readlines():
                    documents_for_test.append(line)
        texts_for_test = ' '.join(documents_for_test)
        with open('./' + category + '/' + category + '.test.txt', 'w') as file:
            file.write(texts_for_test)
        return

    def get_main_article(self, category):
        article = []
        for word in category.lower().split():
            article.append(lmtzr.lemmatize(word))
        main_article = ' '.join(article)
        return self.capitalize_first_letter(main_article)

    def save_corpus_from_textfiles(self):
        documents = []
        for category in self.list_of_categories:
            main_article = self.get_main_article(category) + '.wiki.txt'
            files = os.listdir('./' + category + '/')
            files.remove(main_article)
            count_of_testfiles = math.floor(len(files) / 3)
            random.shuffle(files)
            testfiles = files[:count_of_testfiles]
            self.save_test_files(testfiles, category)
            trainfiles = files[count_of_testfiles:]
            trainfiles.append(main_article)
            for file_name in trainfiles:
                with open('./' + category + '/' + file_name, 'r') as file:
                    for line in file.readlines():
                        documents.append(line)
        texts = [[lmtzr.lemmatize(word) for word in re.findall('\w+', document.lower()) if self.is_word_correct(word)]
                 for document in documents]
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1
        texts = [[token for token in text if frequency[token] > 3]
                 for text in texts]
        dictionary = corpora.Dictionary(texts)
        dictionary.save_as_text(corpus_name + '.dict')
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpora.UciCorpus.save_corpus(
            corpus_name + '.txt', corpus, id2word=dictionary)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", help="Максимальное количество "
                        "файлов для сохранения к каждой теме")
    args = parser.parse_args()
    if args.max:
        wiki = WikiText(categories, args.max)
    else:
        wiki = WikiText(categories)
    wiki.get_textfiles_from_wiki()
    wiki.save_corpus_from_textfiles()
    print('UCI корпус для обучения успешно скачан!\n Для обучения LDA модели -'
          ' используйте комманду "python3 trainmodel.py --count COUNT"')
