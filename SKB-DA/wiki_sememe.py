import numpy as np
import json
from nltk.corpus import wordnet as wn
from math import log
import glob
import tqdm
import json


sememe_entity_sence = np.load("./data/sememe_entity_sence.npy", allow_pickle=True).tolist()
title_content_lemmatization_sence = np.load("./data/title_content_lemmatization_sence.npy", allow_pickle=True).tolist()


def _getTextFile(langual):
    file_list = glob.glob(f'../data/stopwords/stopwords/*_{langual}.txt')
    files = ",".join(file_list)
    return files

stop_words=set()
for file in _getTextFile("en").split(","):
    for word in open(file):
        stop_words.add(word.strip())

title_list = list(title_content_lemmatization_sence.keys())
stop_words.add("")
stop_words.add("'s")
stop_words.add("States")
stop_words.add("United")

cdv = np.load("./sememe_network_cdv_en_wordnet_2000.npy", allow_pickle=True).tolist()

tagme_sememe_dict_l = {}
for items in tqdm.tqdm(sememe_entity_sence):
    if not items:
        continue 
    i, entityList = items.split("\t")
    if entityList == "null" or len(entityList) == 0:
        continue
    entityList = json.loads(entityList)
    entities = [d['spot'] for d in entityList if 'title' in d and float(d['rho']) > 0.1]
    #entities = [d['title'] for d in entityList if 'title' in d and float(d['rho']) > 0.1]
    title = title_list[int(i)]
    if title not in tagme_sememe_dict_l.keys():
        tagme_sememe_dict_l[title] = []
    entities = " ".join(entities).strip().split(" ")
    tagme_sememe_dict_l[title] = entities

title_sememe_raw = {}
for k, v in tqdm.tqdm(tagme_sememe_dict_l.items()):
    sememe = []
    for w in v:
        if w in cdv:
        #if w in dict_sememes:
            sememe.append(w)
    if len(sememe) == 0:
        continue
    title_sememe_raw[k] = sememe

print("take max [:3000] in content lexion for sememe (raw)")
countMinMaxAver(title_sememe_raw)

sememe_freq = {}
sememe_set = set()
for doc_sememes in title_sememe_raw.values():
    for sememe in doc_sememes:
        sememe_set.add(sememe)
        if sememe in sememe_freq:
            sememe_freq[sememe] += 1
        else:
            sememe_freq[sememe] = 1

sememe_lexion = list(sememe_set)
sememe_lexion_size = len(sememe_lexion)

sememe_id_map = {}
for i in range(sememe_lexion_size):
    sememe_id_map[sememe_lexion[i]] = i

print("sememe lexion size: ", sememe_lexion_size)

doc_word_freq = {}
for doc_id,(_, doc_words) in enumerate(title_sememe_raw.items()):
    for word in doc_words:
        word_id = sememe_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1
            
word_doc_list = {}
for i,(_, doc_words) in enumerate(title_sememe_raw.items()):
    appeared = set()
    for word in doc_words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)
        
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)
    
tfidf_word2doc_all = {}
for i,(title, doc_words) in enumerate(title_sememe_raw.items()):
    doc_word_set = set()
    for word in doc_words:
        if word in doc_word_set or word == title:
            continue
        j = sememe_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        idf = log(1.0 * len(title_sememe_raw) / word_doc_freq[sememe_lexion[j]])
        tfidf_word2doc_all[key] = freq * idf
        doc_word_set.add(word)

title_list = [word for word in title_sememe_raw.keys()]
tfidf_filter = {}
for title, tfidf in tfidf_word2doc_all.items():
    if tfidf > 4:
        w, d = title.split(",")
        if title_list[int(w)] not in tfidf_filter.keys():
            tfidf_filter[title_list[int(w)]] = []
        tfidf_filter[title_list[int(w)]].append(sememe_lexion[int(d)])
print("all lexion size:", len(tfidf_filter))

# this part for add pmi weight to building graph
#title_list = [word for word in title_sememe_raw.keys()]
#tfidf_filter = {}
#for title, tfidf in tfidf_word2doc_all.items():
#    if tfidf > 4:
#        w, d = title.split(",")
#        if title_list[int(w)] not in tfidf_filter.keys():
#            tfidf_filter[title_list[int(w)]] = []
#        pmi_weight = sememe_lexion[int(d)]+":"+str(tfidf)
#        tfidf_filter[title_list[int(w)]].append(pmi_weight)
#print("all lexion size:", len(tfidf_filter))
#tfidf_filter = {k:v for k, v in tfidf_filter.items() if k in sememe_network.keys()}
#print("all lexion size:", len(tfidf_filter))
#np.save("wiki_sememe_423249_212", tfidf_filter)

tagme_sememe_dict = {}
for items in tqdm.tqdm(sememe_entity_sence):
    if not items:
        continue 
    i, entityList = items.split("\t")
    if entityList == "null" or len(entityList) == 0:
        continue
    entityList = json.loads(entityList)
    #entities = [d['spot'] for d in entityList if 'title' in d and float(d['rho']) > 0.1]
    entities = [d['title']+"#"+str(d['link_probability']) for d in entityList if 'title' in d and float(d['rho']) > 0.01]
    #print(i)
    title = title_list[int(i)]
    if title not in tagme_sememe_dict.keys():
        tagme_sememe_dict[title] = []
    tagme_sememe_dict[title] = entities

title_list = list(title_content_lemmatization_sence.keys())
max_value = 4
sememe_network = {}
for word, sememes in tfidf_filter.items():
    if len(sememes) != 0:
        if len(sememes) > max_value:
            tagme_sememe_dict_v = tagme_sememe_dict[word]
            sememe_freq_max_value = {}
            for sememe in sememes:
                for v in tagme_sememe_dict_v:
                    #print(v)
                    s, l = v.split("#")
                    if s == sememe:
                        sememe_freq_max_value[sememe] = float(l)
            sememe_lower = sorted(sememe_freq_max_value.items(),key=lambda item:item[1],reverse=True)[:max_value]
            sememe_lower = [sememe_set[0] for sememe_set in sememe_lower]
            sememe_network[word] = sememe_lower
        else:
            sememe_network[word] = sememes
    else:
        continue

print("all lexion size:", len(sememe_network))
sum([len(v) for v in sememe_network.values()])/len(sememe_network)
word_freq = {}
word_set = set()
for words in sememe_network.values():
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

print("vocab_size: ", vocab_size)
sememe_network = {k:v for k, v in sememe_network.items() if len(v) != 0}
np.save("sememe_network_dict_en_wiki_2000",sememe_network)
np.save("sememe_network_cdv_en_wiki_2000",vocab)


