#!/usr/bin/env python
# coding: utf-8

# In[1]:


#robots.txt rulet to check which URL crawler can access on websites
import requests
requestResponseOxford = requests.get("https://www.ox.ac.uk/robots.txt")
requestResponseCambridge = requests.get("https://www.cam.ac.uk/robots.txt")
requestResponseAndrews = requests.get("https://www.st-andrews.ac.uk/robots.txt")

restrictionCheckOxford = requestResponseOxford.text
restrictionCheckCambridge = requestResponseCambridge.text
restrictionCheckAndrews = requestResponseAndrews.text

print("robots.text file to check Oxford University")
print("-------------------------------------------------","\n",restrictionCheckOxford)
print("\n","robots.text file to check Cambridge University")
print("-------------------------------------------------","\n",restrictionCheckCambridge)
print("\n","robots.text file to check St Andrews University")
print("-------------------------------------------------","\n",restrictionCheckAndrews)


# In[2]:


#import required libraries
import re
from bs4 import BeautifulSoup
from pandas import DataFrame
import pandas as pd
from IPython.display import display


#Declared list of empty list to create dataframe 
uniNames = []
guardianScore = []
uniLinks = []
courseTitle = []
aboutCourse = []
assessmentType = []
courseLinks = []
duration = []
country = []
stateOfEngland="England"
stateOfScotland="Scotland"
uniNameOxf = "University of Oxford"
uniNameCam = "University of Cambridge"
uniNameAndrew = "University of St Andrews"
guardianScoreOxf = "95.4" 
guardianScoreCam = "98.6"
guardianScoreAndrew = "100"
finalDf = pd.DataFrame()


# In[4]:


#Function to crawl data
def crawler(url1,url2,url3):
    getDetailsFromOxford(url1)
    getDetailsFromCambridge(url2)
    getDetailsFromAndrews(url3)
    data={"UniversityName": uniNames, "GuardianScore": guardianScore, "UniversityLink": uniLinks, "Duration": duration,
          "CourseTitle": courseTitle, "ShortDescription": aboutCourse, "AssessmentType":assessmentType, "CourseLink": courseLinks, "Country": country}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    return df

#Function to fetch oxford uni details
def getDetailsFromOxford(Url):
    text = requests.get(Url).text
    soup = BeautifulSoup(text)
    domain = 'https://www.ox.ac.uk'
    for x,y in zip(soup.findAll('div', {'class':'course-title'}),soup.findAll('div', {'class':'course-duration course-cell'})):
        degreeTitle = x.text
        degreeLink=x.select_one('a').get('href')
        print(degreeTitle)
        print(domain+degreeLink)
        courseTitle.append(degreeTitle)
        courseLinks.append(domain+degreeLink)
        year = y.text
        print(year)
        duration.append(year)
        guardianScore.append(guardianScoreOxf)
        country.append(stateOfEngland)
        uniLinks.append(domain)
        uniNames.append(uniNameOxf)
        getCourseDetailsOxford(domain+degreeLink)
        
def getCourseDetailsOxford(url):
    text = requests.get(url).text
    soup = BeautifulSoup(text)
    about = soup.find('div', {'class':'field-name-field-intro'})
    desc = about.select_one("span").text
    print(desc)
    aboutCourse.append(desc)
    t=[]
    for header in soup.findAll('h3'):
        if 'Assessment' in header.text:
            for elem in header.next_siblings:
                if elem.name == 'h3':
                    break
                if elem.name != 'p':
                    continue
                print(elem.text)
                t.append(elem.text)
            listToString=' '
            for ass in t:
                listToString += ' '+ass
            assessmentType.append(listToString)
                
    
#Function to fetch cambridge uni details
def getDetailsFromCambridge(Url):
    text = requests.get(Url).text
    soup = BeautifulSoup(text)
    domain = "https://www.cam.ac.uk"
    for z,p in zip(soup.findAll('h4',{'class':'panel-title'}),soup.findAll('a',{'class':'campl-primary-cta'})):
        degreeTitle=z.text
        degreeLink=p.get('href')
        print(degreeTitle)
        print(degreeLink)
        courseTitle.append(degreeTitle)
        courseLinks.append(degreeLink)
        country.append(stateOfEngland)
        guardianScore.append(guardianScoreCam)
        uniLinks.append(domain)
        uniNames.append(uniNameCam)
        getLengthOfCourse(degreeLink)
        getCourseDetailsCambridge(degreeLink)
    
def getCourseDetailsCambridge(url):
    text = requests.get(url+'/study').text
    soup = BeautifulSoup(text)
    about=[]
    assessment=[]
    for header in soup.findAll('h1'):
        if 'Teaching' in header.text:
            for elem in header.next_siblings:
                if elem.name == 'h1' or elem.name == 'h2':
                    break
                if elem.name != 'p':
                    continue
                print(elem.text)
                about.append(elem.text)
            listToString=' '
            for desc in about:
                listToString += ' '+desc
            aboutCourse.append(listToString)
    for header in soup.findAll('h2'):
        if 'Thesis / Dissertation' in header.text:
            for elem in header.next_siblings:
                if elem.name == 'h2':
                    break
                if elem.name != 'p':
                    continue
                print(elem.text)
                assessment.append(elem.text)
            lString=' '
            for ass in assessment:
                lString += ' '+ass
            assessmentType.append(lString)
            
            
#Function to fetch StAndrews uni details
def getDetailsFromAndrews(Url):
    text = requests.get(Url).text
    soup = BeautifulSoup(text)
    domain = "https://www.st-andrews.ac.uk"
    for link in soup.findAll('a',{'class':'list-group-item'}):
        degreeTitle=link.select_one('span').text
        degreeLink=link.get('href')
        print(degreeTitle)
        print(degreeLink)
        courseTitle.append(degreeTitle)
        courseLinks.append(degreeLink)
        country.append(stateOfScotland)
        guardianScore.append(guardianScoreAndrew)
        uniLinks.append(domain)
        uniNames.append(uniNameAndrew)
        duration.append("1 year full-time")
        getCourseDetailsAndrews(degreeLink)
        
def getCourseDetailsAndrews(url):
    text=requests.get(url).text
    soup=BeautifulSoup(text)
    about = soup.find('div', {'class':'page-intro__text'})
    desc=about.findAll('p')
    about=[]
    assessment=[]
    for x in desc:
        print(x.text)
        about.append(x.text)
    listToString=' '
    for desc in about:
        listToString += ' '+desc
    aboutCourse.append(listToString)
    atype=soup.find('div',{'id':'tab-12'})
    Type= atype.findAll('p')
    for y in Type:
        print(y.text)
        assessment.append(y.text)
    lString=' '
    for ass in assessment:
        lString += ' '+ass
    assessmentType.append(lString)
        
#Functions to fetch length of course
def getLengthOfCourse(degreeUrl):
    text = requests.get(degreeUrl).text
    soup = BeautifulSoup(text)
    length = soup.select_one('h4').text
    print(length)
    duration.append(length)    
    
#Function to get crawled data
def getData():
    url1 = 'https://www.ox.ac.uk/admissions/graduate/courses/mpls/computer-science'
    url2 = 'https://www.postgraduate.study.cam.ac.uk/courses/departments/cscs?_gl=1*8g1fbe*_ga*OTYyODM0NDk1LjE2Njc4MzcyMzU.*_ga_P8Q1QT5W4K*MTY2ODEyNzAyNC45LjEuMTY2ODEyNzA2OS4wLjAuMA..'
    url3 = 'https://www.st-andrews.ac.uk/subjects/computer-science/'
    text = requests.get(url1).text
    soup = BeautifulSoup(text)
    df1 = crawler(url1,url2,url3)
    display(df1)
    return df1

#Final data frame of the crawled data
finalDf = getData()


# In[7]:


#Display function to view the final data frame
display(finalDf)
finalDf.to_csv("C:/Users/giric/OneDrive/Desktop/smartengine.csv", sep=',',index=False)


# In[8]:


#Importing the required libraries
import glob 
import math
import re
import os
import numpy as np
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize

#Stopwords & Punctuations
Stopwords = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,\n<>./?@#$%^&*_~'''

#Read the corpus as CSV file
data = pd.read_csv('C:/Users/giric/OneDrive/Desktop/smartengine.csv')
    
# pre processing the data 
data = data.fillna(" ")
data['documents'] = data["UniversityName"] + " " + data["CourseTitle"]+data['ShortDescription']

# making the documents to lower case
data['documents'] = data['documents'].str.lower()
docum = 'docums'
docs = data['documents'].tolist()
count = 1;
Documents_dict = {}
dict_vocab = {} 
ps = PorterStemmer()
filtered_docs = []

#Tokenization & Stopword removal
for doc in docs:
    tokens = word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in Stopwords:
            tmp += ps.stem(w) + " "
    filtered_docs.append(tmp)

# Removal of punctuations
for file in filtered_docs:     
    text = file.replace('\n', ' ')
    final_text = ""
    for char in text:
        if char not in punctuations:
            final_text = final_text + char
    list_word_text = final_text.split(" ")
    
#Storing the frequency of words
    for word in list_word_text:
        if word in dict_vocab.keys():
            dict_vocab[word] = dict_vocab[word] + 1
        else:
            dict_vocab[word] = 1
    Length_text = len(list_word_text)
    uniq_set = set(list_word_text)
    uniq_text = (list(uniq_set))
    print("Document - ", count, ":")
    print(final_text, "\n")
    
#Term frequency & Norm Term frequency
    TF_dict = {}
    print("TERMS", "->", "TERM FREQ", "->", "NORM TERM FREQ  (term freq/ doc size) ", "\n")
    doc_norm = 0;
    for words in uniq_text:
        print(words, " -> ", list_word_text.count(words), " -> ", list_word_text.count(words), "/",Length_text)
        temp_list = [list_word_text.count(words), list_word_text.count(words)/Length_text]
        TF_dict[words] = temp_list
        doc_norm = doc_norm + TF_dict[words][1]*TF_dict[words][1]
    doc_normc = math.sqrt(doc_norm)
    TF_dict["doc_norm"+str(count)] = doc_normc
    print("\n|| d"+str(count)+" || = ", doc_normc)
    Documents_dict["doc"+str(count)] = TF_dict
    print("\n")
    print("TF_dict",TF_dict)
    count = count + 1
freq_vocab = []
uniq_vocab = []
for word in dict_vocab.keys():
    if dict_vocab[word] > 10:
        freq_vocab.append(word)
    elif dict_vocab[word] == 1:
        uniq_vocab.append(word)
print("High frequency words :")
print(freq_vocab, "\n")
print("Rare words:")
print(uniq_vocab, "\n")
count = count - 1

# other way of finding inverted index using enumerator function
dict_inverted_index ={}
# inverted index
for i, doc in enumerate(filtered_docs):
    for word in doc.split(): 
        if word in dict_inverted_index:
            dict_inverted_index[word].add(i) 
        else:
            dict_inverted_index[word] = {i} 
print(dict_inverted_index)


# In[9]:


print(dict_inverted_index)


# In[10]:


Documents_dict


# In[11]:


import string
print("Enter your query: ")
query = input("Please input your query: ")
#query = query.split(" ")
query_set = set(query)

# Data preprocessing
query_list = []
query_lowercase = query.lower()

# punctuation removal
new_string = query_lowercase.translate(str.maketrans('', '', string.punctuation))

#query = word_tokenize(new_string)
tokens = word_tokenize(new_string)
print(tokens)   
tmp = ""
for w in tokens:
    u = w.lower()
    if u not in Stopwords:
        tmp = ps.stem(w)
        x=re.sub(r"\s+$", "", tmp)
    query_list.append(x)
print("query_list after pre processing", query_list)
i=0
Query_doc_freq = {}
Total_doc_freq = 0
while i<len(query_list):
    doc_freq = 0
    j=1
    while j<=count:
        if query_list[i] in Documents_dict["doc"+str(j)].keys():
            doc_freq = doc_freq + 1
        j=j+1
    Total_doc_freq = Total_doc_freq + doc_freq
    Query_doc_freq[query_list[i]] = doc_freq
    i = i+1
if Total_doc_freq == 0:
    print("--------Query not found--------")
else:
    print("\n\nTERMS", "->", "DOC FREQ", "->", "NORM DOC FREQ  (doc freq/total doc freq)\n")
    norm_q = 0
    for Keyss in Query_doc_freq:
        print(Keyss, "->", Query_doc_freq[Keyss], "->", Query_doc_freq[Keyss]/Total_doc_freq)
        # Similarity of documents using the cosine function
        # vec1[i]*vec2[i]/length(vec1)*length(vec2)
        Query_doc_freq[Keyss] = Query_doc_freq[Keyss]/Total_doc_freq
        norm_q = norm_q + Query_doc_freq[Keyss]*Query_doc_freq[Keyss]
    norm_qq = math.sqrt(norm_q)
    print("\n|| q || = ", norm_qq, "\n")
    
    Similarity_scores = {}
    i=1
    while i<= count:
        sim_temp = 0
        j=0
        while j<len(query_list):
            if query_list[j] in Documents_dict["doc"+str(i)].keys():
                sim_temp = sim_temp + Documents_dict["doc"+str(i)][query_list[j]][1]*Query_doc_freq[query_list[j]]
            j=j+1
        denominator = norm_qq*Documents_dict["doc"+str(i)]["doc_norm"+str(i)]
        Similarity_scores["doc"+str(i)] = sim_temp/denominator
        i=i+1
        
    print("Similarity scores:")
    print(Similarity_scores)
    print("\nRanking based on similarity scores: ")
    print (sorted(Similarity_scores.items(), reverse=True, key = lambda x : x[1]))
    print("\n")


# In[12]:


z= sorted(Similarity_scores.items(), reverse=True, key = lambda x : x[1])


# In[14]:


#How-to-extract-the-substring-between-two-markers
import ast
punctuations_list ='''!()[]{};'"\,\n<>.?@#$%^&*_~'''
#https://www.geeksforgeeks.org/python-convert-a-string-representation-of-list-into-list/
for x in z:
    if x[1]>0.0 :
        num = ""
        for c in x[0]:
            if c.isdigit():
                num = num + c
        d = int(num)-1
        si =int(d)
        format(si)
        
def format(a):
    print(f"University Name: {data.UniversityName.iloc[a]}")
    print("")
    print(f"UniLink: {data.UniversityLink.iloc[a]}")
    print("")
    print(f"The Guardian Score: {data.GuardianScore.iloc[a]}")
    print("")
    print(f"Course Name: {data.CourseTitle.iloc[a]}")   
    print("       ")
    print(f"About Course: {data.ShortDescription.iloc[a]}")
    print("")
    print(f"Course Link: {data.CourseLink.iloc[a].split()[-1]}")
    print("")
    print(f"Country: {data.Country.iloc[a].split()[-1]}")
    print("")
    print(f"_____________________________________________________________________________________")
    print("")




