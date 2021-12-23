import numpy as np
import json, os
from nltk.corpus import wordnet as wn
from math import log
import glob
import tqdm
import json
import spacy
import matplotlib.pyplot as plt

def _getTextFile(langual):
    file_list = glob.glob(f'../data/stopwords/stopwords/*_{langual}.txt')
    files = ",".join(file_list)
    return files

def cleanText(english_txt):
    try:
        word_tokens = english_txt.split()
        filtered_word = [w for w in word_tokens if w not in stop_words and not w.isdigit()]
        filtered_word = [w + " " for w in filtered_word]
        return "".join(filtered_word)
    except:
        return np.nan

def detectLang(txt):
    try:
        return detect(txt)
    except:
        return np.nan

def cleanNonEnglish(txt):
    txt = re.sub(r'\W+', ' ', txt)
    txt = txt.lower()
    txt = txt.replace("[^a-zA-Z]", " ")
    word_tokens = txt.split()
    filtered_word = [w for w in word_tokens if all(ord(c) < 128 for c in w)]
    filtered_word = [w + " " for w in filtered_word]
    return "".join(filtered_word)

def countMinMaxAver(lines):
    min_len = 10000
    aver_len = 0
    max_len = 0
    for temp in lines:
        aver_len = aver_len + len(temp)
        if len(temp) < min_len:
            min_len = len(temp)
        if len(temp) > max_len:
            max_len = len(temp)
    aver_len = 1.0 * aver_len / len(lines)
    print('min_len : ' + str(min_len))
    print('max_len : ' + str(max_len))
    print('average_len : ' + str(aver_len))
    
def titleProcessing(title):
    #title = re.findall(r"title=\"(.*)\">", title)
    #title = title[0].replace(" ", "_")
    return re.findall(r"title=\"(.*)\">", title)

def contentProcessing(content, title_list_i):
    if content[0].strip() == title_list_i[0]:
        return title_list_i, content[1].strip()
stop_words=set()
for file in _getTextFile("en").split(","):
    for word in open(file):
        stop_words.add(word.strip())

tagme_json = np.load("../data/wordnet_sememe_entity_sence.npy", allow_pickle=True).tolist()
title_content_lemmatization = np.load("../data/wordnet_lemmatization_sence.npy", allow_pickle=True).tolist()


title_list = list(title_content_lemmatization.keys())
stop_words.add("")
stop_words.add("'s")
stop_words.add("States")
stop_words.add("United")

tagme_sememe_dict_l = {}
for items in tqdm.tqdm(tagme_json):
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

tagme_sememe_dict = {k:v for k,v in tagme_sememe_dict_l.items() if len(v)!=0}
for k, v in tagme_sememe_dict.items():
    def_list = []
    for v_i in v:
        if v_i not in stop_words:
            def_list.append(v_i)
    tagme_sememe_dict[k]=def_list
word_freq = {}
word_set = set()
for doc_words in tagme_sememe_dict.values():
    for word in doc_words:
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

print("content lexion: ", vocab_size)
word_freq_sorted = sorted(word_freq.items(), key = lambda kv:(kv[1], kv[0]))
sememe_raw = [word for (word, _) in word_freq_sorted[-2000:]]


title_sememe_raw = {}
for k, v in tqdm.tqdm(tagme_sememe_dict.items()):
    sememe = []
    for w in v:
        if w in sememe_raw:
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
    if tfidf > 0:
        w, d = title.split(",")
        if title_list[int(w)] not in tfidf_filter.keys():
            tfidf_filter[title_list[int(w)]] = []
        tfidf_filter[title_list[int(w)]].append(sememe_lexion[int(d)])
print("all lexion size:", len(tfidf_filter))

# this part for add pmi weight to building graph
title_list = [word for word in title_sememe_raw.keys()]
tfidf_filter = {}
for title, tfidf in tfidf_word2doc_all.items():
    if tfidf > 0:
        w, d = title.split(",")
        if title_list[int(w)] not in tfidf_filter.keys():
            tfidf_filter[title_list[int(w)]] = []
        pmi_weight = sememe_lexion[int(d)]+":"+str(tfidf)
        tfidf_filter[title_list[int(w)]].append(pmi_weight)
print("all lexion size:", len(tfidf_filter))

tfidf_filter = {k:v for k, v in tfidf_filter.items() if k in sememe_network.keys()}
print("all lexion size:", len(tfidf_filter))

sememe_network_pmi = {}
for k,v in tfidf_filter.items():
    for vi in v:
        vi,pmi=vi.split(":")
        if vi in sememe_network[k]:
            if k not in sememe_network_pmi.keys():
                sememe_network_pmi[k] = []
            sememe_network_pmi[k].append(vi+":"+pmi)
np.save("wordnet_sememe_163340_218", sememe_network_pmi)

sememe_network_pmi = {}
for k,v in tfidf_filter.items():
    for vi in v:
        vi,pmi=vi.split(":")
        if vi in sememe_network[k]:
            if k not in sememe_network_pmi.keys():
                sememe_network_pmi[k] = []
            sememe_network_pmi[k].append(vi+":"+pmi)


np.save("wordnet_sememe_163340_218", sememe_network_pmi)

title_id_map = {w:i for i,w in enumerate(title_list)}
max_value = 4
sememe_network = {}
for word, sememes in tfidf_filter.items():
    if len(sememes) != 0:
        if len(sememes) > max_value:
            sememe_freq_max_value = {}
            for sememe in sememes:
                tfidf_word2doc_all_key = str(title_id_map[word]) +","+ str(sememe_id_map[sememe])
                sememe_freq_max_value[sememe] = tfidf_word2doc_all[tfidf_word2doc_all_key]
            sememe_lower = sorted(sememe_freq_max_value.items(),key=lambda item:item[1],reverse=True)[:max_value]
            sememe_lower = [sememe_set[0] for sememe_set in sememe_lower]
            sememe_network[word] = sememe_lower
        else:
            sememe_network[word] = sememes
    else:
        continue




