#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[4]:


import io
df = pd.read_csv('smartengine.csv')


# In[5]:


df.head()


# In[6]:


#Pre-processing for short description
preProcessingDescription = df['ShortDescription']

#Stop word, removing punctuation & stemming the string
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words('english')
print(sw)

nltk.download("punkt")
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
docAsList = []
preProcessedText = []
for description in preProcessingDescription:
    tokens = word_tokenize(description)
    t=[]
    tmp = ""
    for w in tokens:
        if w not in sw and w.isalpha():
            t.append(stemmer.stem(w).lower())
            tmp += stemmer.stem(w) + " "
    docAsList.append(t)
    preProcessedText.append(tmp)

print(docAsList)
print(preProcessedText)


# In[7]:


#Pre-processing for uni name
preProcessingUniName = df['UniversityName']
docAsList1 = []
preProcessedText1 = []
for uniName in preProcessingUniName:
    tokens = word_tokenize(uniName)
    t=[]
    tmp = ""
    for w in tokens:
        if w not in sw and w.isalpha():
            t.append(stemmer.stem(w).lower())
            tmp += stemmer.stem(w) + " "
    docAsList1.append(t)
    preProcessedText1.append(tmp)

print(docAsList1)
print(preProcessedText1)


# In[8]:


#To view first 15 record from data frame after pre-processing
df['NewPreProcessedShortDesc'] = docAsList
df['NewPreProcessedUniName'] = docAsList1
df['PreProcessedTextOfShortDesc'] = preProcessedText
df['PreProcessedTextOfUniName'] = preProcessedText1
a = df[['ShortDescription','NewPreProcessedShortDesc','PreProcessedTextOfShortDesc',
             'UniversityName','NewPreProcessedUniName','PreProcessedTextOfUniName']]
a.head(15)


# In[9]:


#Indexing the text
text = []
for t in df['NewPreProcessedShortDesc']:
    text.append(t)

for txt in text:
    index = {w :[(text.index(txt))+1] for w in txt}
    
index


# In[10]:


#Inverted Index
from collections import defaultdict

class invertedIndex(object):
    def __init__(self,docs):
        self.docSets = defaultdict(set)
        for indx, doc in enumerate(docs):
            for term in doc.split():
                self.docSets[term].add(indx)
        
        print(self.docSets)
        
    def search(self,term):
        return self.docSets[term]
    
docs = [sent for sent in df.PreProcessedTextOfShortDesc]
docs1 = [sent for sent in df.PreProcessedTextOfUniName]
indexed = invertedIndex(docs)
indxed = invertedIndex(docs1)


# In[11]:


get_ipython().system('pip install transformers')
get_ipython().system('pip install sentencepiece')
get_ipython().system('pip install torch')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install tensorflow_hub')
get_ipython().system('pip install tensorflow_text')


# In[12]:


get_ipython().system('pip install tensorflow_text')


# In[13]:


get_ipython().system('pip install -q -U "tensorflow-text==2.8.*"')


# In[14]:


get_ipython().system('pip install -U tensorflow-text')


# In[15]:


# importing libraries
import tensorflow as tf
import tensorflow_hub as hub
# import tensorflow_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import string
import warnings
warnings.filterwarnings("ignore")



# In[16]:


get_ipython().system('pip install bert-for-tf2')


# In[17]:


import bert


# In[18]:


# importing BERT model
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
bert_layer = hub.KerasLayer(module_url, trainable=True)


# In[19]:


# BERT model
def bert_encode(texts, tokenizer, max_len=512):
    print(texts)
    print(tokenizer)
    all_tokens = []
    all_masks = []
    all_segments = []
    for text in texts:
        print(text)
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


# In[20]:


# importing BERT tokenizer
FullTokenizer = bert.bert_tokenization.FullTokenizer
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)


# In[21]:


# encoding data
train_input = bert_encode(df.ShortDescription.values, tokenizer, max_len=160)
train_labels = df.UniversityName.values
train_masks = train_input[1]
train_segments = train_input[2]


# In[22]:


# building model
def build_model(bert_layer, max_len=160):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")
    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[23]:


# training model
model = build_model(bert_layer, max_len=160)
model.summary()
train_history = model.fit(
    [train_input[0], train_masks, train_segments], train_labels,
    validation_split=0.2,
    epochs=3,
    batch_size=16
);


# In[24]:


# saving model
model.save('bert_model.h5')


# In[25]:


# loading model
model = tf.keras.models.load_model('bert_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})


# In[26]:


model


# In[27]:


# checking the model
def check_model(query):
    query = [query]
    print(query)
    query = bert_encode(query, tokenizer, max_len=160)
    query_masks = query[1]
    query_segments = query[2]
    return model.predict([query[0], query_masks, query_segments])

# checking the model
check_model('Software Engineering')


# In[ ]:




