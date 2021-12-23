#!/usr/bin/env python
# coding: utf-8

# In[2]:


from langdetect import detect
import numpy as np
import re
import glob
import os
import tqdm
import nltk
nltk.download('punkt') # one time execution
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[14]:


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


# In[15]:


stop_words=set()
for file in _getTextFile("en").split(","):
    for word in open(file):
        stop_words.add(word.strip())


# In[6]:


index_content_dic = {}
title_list = []
count = 0
for line in open('../data/wiki.txt'):
    if re.match('<doc id.*>', line):
        title_list.append(titleProcessing(line))
        index_content_dic[count] = []
        count+=1
        continue
    index_content_dic_value = index_content_dic[count]
    index_content_dic_value.append(line)
    index_content_dic[count] = index_content_dic_value
    
countMinMaxAver(index_content_dic.values())


# In[19]:


len(index_content_dic)


# In[20]:


len(title_list)


# In[21]:


title_content_dic = {}
for i, content in tqdm.tqdm(index_content_dic.items()):
    title, content = contentProcessing(content, title_list[i])
    content = cleanNonEnglish(content).strip()
    content = cleanText(content).strip()
    title_content_dic[title[0]] = content


# In[22]:


list(title_content_dic.items())[:10]


# In[24]:


np.save("../data/title_content_dic", title_content_dic)


# In[ ]:




