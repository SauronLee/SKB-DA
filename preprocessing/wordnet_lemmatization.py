wordnet_tagme_sememe_dict: dict = np.load("../data/wordnet_tagme_sememe_dict.npy", allow_pickle=True).tolist()

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
    
def lemmatization(sentence):
    tokens = word_tokenize(sentence)
    tagged_sent = pos_tag(tokens)
    wnl = WordNetLemmatizer()
    lemmas_sent = []
    for tag in tagged_sent:
        wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
        lemmas_sent.append(wnl.lemmatize(tag[0], pos = wordnet_pos))
    return lemmas_sent

title_content_lemmatization = {}
for word, doc in tqdm.tqdm(wordnet_tagme_sememe_dict.items()):
    doc = lemmatization(doc)
    title_content_lemmatization[word] = doc

np.save("../data/wordnet_lemmatization_sence", title_content_lemmatization)
