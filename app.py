
from flask import Flask, render_template, request;
import requests;
import re
from bs4 import BeautifulSoup
from pandas import DataFrame
import pandas as pd
from IPython.display import display
import math
import re
import os
import numpy as np
import sys
import pandas as pd
import nltk
import string
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import sent_tokenize , word_tokenize
# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)

# Flask route decorators map / and /hello to the hello function.
# To add other resources, create functions that generate the page contents
# and add decorators to define the appropriate resource locators for them.


def robotTxt():
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


@app.route('/')
@app.route('/smartengine')
def smartEngine():
    robotTxt()
    finalDf = getData()
    return render_template('search.html')


#Display function to view the final data frame
display(finalDf)
finalDf.to_csv("C:/Users/giric/OneDrive/Desktop/smartengine.csv", sep=',',index=False)

#Stopwords & Punctuations
Stopwords = set(stopwords.words('english'))
punctuations = '''!()-[]{};:'"\,\n<>./?@#$%^&*_~'''

#Read the corpus as CSV file
#data = pd.read_csv('C:/Users/giric/OneDrive/Desktop/smartengine.csv')

@app.route('/', methods=['POST'])
def getSearchTerm():
    query = request.form['searchTerm']
    #test = retrieveData(query)
    return render_template('result.html', q=query)      

if __name__ == '__main__':
    # Run the app server on localhost:4449
    app.run('localhost', 4449)