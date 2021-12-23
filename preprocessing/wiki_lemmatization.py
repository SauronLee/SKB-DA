#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import log
import tqdm
import json
import numpy as np
import spacy
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

title_content_dic: dict = np.load("../data/sememe_entity_sence.npy", allow_pickle=True).tolist()

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def lemmatization(sentence, title):
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)
    title_p = "NN"
    for w, p in tagged_sent:
        if w == title:
            title_p = p
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos = wordnet_pos))
    return lemmas_sent, title_p

title_content_lemmatization = {}
for word, doc in tqdm.tqdm(title_content_dic.items()):
    word_e = word.strip().split()[0]
    doc, word_p = lemmatization(doc, word_e)
    title_content_lemmatization[word+">>>"+word_p] = doc

np.save("../data/title_content_lemmatization_sence", title_content_lemmatization)

