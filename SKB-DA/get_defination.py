from langdetect import detect
import numpy as np
import re
import glob
import os
import tqdm
import nltk
from nltk.corpus import wordnet as wn
nltk.download('punkt') # one time execution

def _getTextFile(langual):
    file_list = glob.glob(f'./data/stopwords/*_{langual}.txt')
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
# get stopwords
stop_words=set()
for file in _getTextFile("en").split(","):
    for word in open(file):
        stop_words.add(word.strip())

# get wikipedia dataset
index_content_dic = {}
title_list = []
count = -1
for line in open('./data/wiki.txt'):
    if re.match('<doc id.*>', line):
        title_list.append(titleProcessing(line))
        count+=1
        index_content_dic[count] = []
        continue
    index_content_dic_value = index_content_dic[count]
    index_content_dic_value.append(line)
    index_content_dic[count] = index_content_dic_value

# countMinMaxAver(index_content_dic.values())
# min_len : 2
# max_len : 5876
# average_len : 8.55975541655267

# len(title_list)
# 6403704

print("Total number of Wikipedia entries: ", len(title_list))

key_word = ["film","novel","album","song","band","name","ep","game","surname","tv series"]

title_content_dic = {}
for i, content in tqdm.tqdm(index_content_dic.items()):
    title = title_list[i][0].strip().split()
    if len(title) == 1 or str(title[1])[0] == "(" and str(title[-1])[-1] == ")":
        if str(title[-1])[:-1].replace("(","") in key_word:
            continue
        title, content = contentProcessing(content, title_list[i])
        content = cleanNonEnglish(content).strip()
        content = cleanText(content).strip()
        if len(content) == 0 or title[0].strip()[0] == "(":
            continue
        title_content_dic[title[0]] = content

print("Filter out complex combinations of words: ", len(title_content_dic))

for k,v in title_content_dic.items():
    if "apple" == k.strip().split()[0]:
        print(k,":---:",v)
        print("----------------------------")

for k, v in title_content_dic.items():
    v_list = []
    for v_i in v.split():
        if v_i == k:
            continue
        else:
            v_list.append(v_i)
    title_content_dic[k] = " ".join(v_list)

np.save("./data/title_content_dic_sence", title_content_dic)

wn_pos_list = [wn.ADJ,wn.VERB,wn.NOUN,wn.ADV]
wn_words = set(i for i in wn.words())
wordnet_tagme_sememe_dict = {}
for word in wn_words:
    for pos in wn_pos_list:
        for i,synset in enumerate(wn.synsets(word, pos=pos)):
            title = word+"."+pos+"."+str(i)
            definition = synset.definition()
            if title not in wordnet_tagme_sememe_dict.keys():
                wordnet_tagme_sememe_dict[title] = []
            wordnet_tagme_sememe_dict[title] = definition
            
np.save("./data/wordnet_tagme_sememe_dict", wordnet_tagme_sememe_dict)
print("Total number of wordnet words: ", len(wordnet_tagme_sememe_dict))