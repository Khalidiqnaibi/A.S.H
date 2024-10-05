# required modules
import random
import json
import pickle
import spacy
import numpy as np
import nltk
import string
import requests
import pyttsx3
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import sys
import os
import pymongo
import webbrowser
from googleapiclient.discovery import build
from datetime import timedelta,datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Ash attempt num 4  

user="khalid afif sami iqnaibi"

load_dotenv()
YOU_API_KEY = os.getenv("YOU_API_KEY")    
W_API_KEY = os.getenv("W_API_KEY")

client = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = client["knowledge_db"]
people = mydb["people"]
activities= mydb["activities"]
actlogs=mydb['activities_logs']
changes=mydb["changes"]
organisms=mydb["organisms"]
known_things=mydb["specific_known_things"]
animals=mydb["animals"]
products=mydb["products"]
events=mydb['events']
Qs=mydb["questions"]
new=mydb["new"]
plants=mydb['plants']
relationships=mydb['relationships']
dairy=mydb["dairy"]
weather = mydb["weather"]
forecast = mydb["daily_forecast"]
inputlog=mydb["inputlog"]
chatlog=mydb['chatlog']


class peopledb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newperson"})
        del c["_id"]
        self.newPerson=c
    def addperson(self,catagory,value):
        person=self.newPerson
        person.update({catagory:value})
        people.insert_one(person)
        ch={"change":f"added a person with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getperson(self,param,val):
            person=people.find_one({param: val})
            return person

    def UpdateOnePerson(self,param,val,catagory,newvalue):
        per=self.getperson(param, val)
        ch={"change":f"updated actionlog with catagory: {catagory} at the value: {newvalue} for the person:{per['name']}, old values are: ({catagory},{per[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        per.update({catagory: newvalue})
        people.find_one_and_replace({param: val}, per)
        
    def delPerson(self,catagory,value):
        people.find_one_and_delete({catagory:value})
        ch={"change":f"deleted a person with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def UpdatenewPerson(self,catagory,newvalue):
        newPerson=self.newPerson
        ch={"change":f"updated newPerson with the catagory: {catagory} at the value: {newPerson} for : (newPerson), old values are: ({catagory},{newPerson[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newPerson.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newPerson"},newPerson)

class animalsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newanimal"})
        del c["_id"]
        self.newAnimal=c
    
    def addanimal(self,catagory,value):
        anml=self.newAnimal
        if catagory in["name"]:
            catagory="common name"
        else:
            pass
        anml.update({catagory:value})
        animals.insert_one(anml)
        ch={"change":f"added an new animal with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def delAnimal(self,catagory,value):
        animals.find_one_and_delete({catagory:value})
        ch={"change":f"deleted an animal with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getanimals(self,param,val):
            organism=organisms.find_one({param: val})
            return organism
    
    def UpdateOneanimals(self,param,val,catagory,newvalue):
        animal=self.getorganism(param, val)
        ch={"change":f"updated animals with the catagory: {catagory} at the value: {newvalue} for the animal: ({animal['common name']}), old values are: ({catagory},{animal[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        animal.update({catagory: newvalue})
        animals.find_one_and_replace({param: val}, animal)
        
    def UpdatenewAnimals(self,catagory,newvalue):
        newAnimal=self.newAnimal
        ch={"change":f"updated newAnimal with the catagory: {catagory} at the value: {newvalue} for : (newAnimal), old values are: ({catagory},{newAnimal[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newAnimal.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newAnimal"},newAnimal)

class qdb():
    def __init__(self,):
        self.collection = mydb["questions"]

    def add_question(self, question, answer, answer_from):
        question_data = {
            'question': question,
            'answer': answer,
            "who gave the answer":answer_from
        }
        self.collection.insert_one(question_data)
        ch={"change":f"added the question {question} With the answer : {answer}","time":datetime.now()}
        changes.insert_one(ch)
        print('Question added successfully!')

    def edit_question(self, question, new_answer):
        query = {'question': question}
        answer=self.collection.find_one(query)['answer']
        new_data = {'$set': {'answer': new_answer}}
        ch={"change":f"edited the question {question} With the new answer : {new_answer}, old answer is : {answer}","time":datetime.now()}
        changes.insert_one(ch)
        self.collection.update_one(query, new_data)
        print('Question updated successfully!')

    def delete_question(self, question):
        query = {'question': question}
        answer=self.collection.find_one(query)['answer']
        self.collection.delete_one(query)
        ch={"change":f"deleted the question {question} With the answer : {answer}","time":datetime.now()}
        changes.insert_one(ch)
        print('Question deleted successfully!')

    def get_question(self, question):
        """Get the answer to a question from the database, or search for the answer if not found."""
        query = {'question': question}
        question_data = self.collection.find_one(query)

        if question_data:
            return question_data['answer']
        else:
            print('Question not found in the database. Searching for the answer...')
            answer = self.googlit(question)
            if answer[1]=='result was not found!':
                n=input("what is the right answer?")
                if n in ["idk",'i dont know','i do not now','Idk',"IDK","I do not know","I dont know"]:
                    return None
                else:
                    self.add_question(question, n,f"{user}")
                    return n
            else:
                an=input(f"is {answer[1]} the correct answer?")
                if an in ["yah",'yes','of course','yup','yah','ya','yee','ye','idk','i dont know','i do not know']:
                    self.add_question(question, answer[1],answer[2])
                    return answer[1]
                else:
                    n=input("what is the right answer?")
                    if n in["idk",'i dont know','i do not now','Idk',"IDK","I do not know","I dont know"]:
                        return None
                    else:
                        self.add_question(question, n,f"{user}")
                        return n

    def googlit(self,question):
        res=[]
        po=[]
        result=None
        wikres=None
        url= f"https://www.google.com/search?q={question}"
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 OPR/97.0.0.0"}
        page = requests.get(url,headers=headers)
        soup=BeautifulSoup(page.content,"html.parser")
        k=soup.find(class_="Z0LcW t2b5Cf")
        m=soup.find("b")
        kl=soup.find('div',{'class': "IZ6rdc"})
        g=soup.find('div',{'class': "dDoNo vrBOv vk_bk"})
        l=soup.find('h2',{'class': 'qrShPb kno-ecr-pt PZPZlf q8U8x'})
        lis=soup.find_all('div',{'class': "bVj5Zb FozYP"})
        los=soup.find_all('div',{'class': "WGwSK ghJsNe"})
        v=soup.find('div',{'class': "wwUB2c PZPZlf E75vKf"})
        
        if k:
            result=k.get_text()
        elif g:
            result=g.get_text()
        elif kl:
            result=kl.get_text()
        elif lis:
            for i in lis:
                po.append(i.get_text())
            result=None
        elif los:
            for i in los:
                po.append(i.get_text())
            result=None
        
        elif l :
            result=l.get_text()
        elif m:
            result=m.get_text()
        else:
            result=None
        if v:
            morres=v.get_text()
        res.append(question)
        first_result = soup.find("div", {"class": "yuRUbf"}).a["href"]
        

        if "wikipedia.org/wiki/"in first_result:
            result_response = requests.get(first_result)
            result_soup = BeautifulSoup(result_response.text, 'html.parser')
            y = result_soup.find('p')
            if y:
                h=y.find_next_sibling('p').b
                if h:
                    wikres = h.get_text()
                else:
                    wikres=None
            else:
                wikres=None
        else:
            wikres=None
        
        if result:
            if "when" in question and "founded"in question:
                res.append(result.split(".")[0].split(",")[0]+result.split(".")[0].split(",")[1])
            else:
                res.append(sent_tokenize(result)[0])
        elif po:
            res.append(po)
        elif wikres:
            res.append(wikres)
        else:
            res.append("result was not found!")
        res.append(first_result)
        #res=['question','answer','herf']
        return res

    def close_connection(self,):
        """Close the MongoDB connection."""
        client.close()

user_loc=peopledb().getperson("name", "khalid afif sami iqnaibi")["location"]
# Initialize the TTS engine
engine = pyttsx3.init()
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)
engine.setProperty('rate', 225)  # Speed of speech
lemmatizer = WordNetLemmatizer()
# loading the files we made previously
with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','r') )as f:
    emos=json.load(f)
emowords = pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emowords.pkl', 'rb'))
emoclasses = pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emoclasses.pkl', 'rb'))
emomodel = load_model('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/chatbotemo.h5')

with (open("C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/intents.json",'r') )as f:
    ints=json.load(f)
    
with (open("C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/command.json",'r') )as f:
    cmnds=json.load(f)
cmndwords =  pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/cmndwords.pkl', 'rb'))
cmndclasses = pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/cmndclasses.pkl', 'rb'))
cmndmodle = load_model('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/chatbotcmnd.h5')

with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/comm.json','r') )as f:
    comms=json.load(f)
commwords =  pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/commwords.pkl', 'rb'))
commclasses = pickle.load(open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/commclasses.pkl', 'rb'))
commmodle = load_model('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/chatbotcomms.h5')

lrn=''
nlp = spacy.load('en_core_web_md')
propapilty=float (0.001)
emotion='nutral'
ha=100
sa=0
an=0
sc=0
dis=0
tird=0
awk=0
brd=0
emb=0
grt=15
  
def feels(happy=0,sad=0,angry=0,sceared=0,discusted=0,tiredness=0,awkwardness=0,boredom=0,embressed=0,greatful=0):
    ha=ha+happy
    sa=sa+sad
    an=an+angry
    sc=sc+sceared
    dis+dis+discusted
    tird=tird+tiredness
    awk=awk+awkwardness
    brd=brd+boredom
    emb=emb+embressed
    grt=grt+greatful
    print(ha,sa,an,sc,dis,tird,awk,brd,emb,grt)    
         
def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) 
                      for word in sentence_words]
    return sentence_words
# Preprocess the text data
def preprocess(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens
# Extract the features from the text data
def extract_features(text,fdist):
    words = set(text)
    features = {}
    for word in fdist.keys():
        features[word] = (word in words)
    return features

def extract_qustion(text):
    
    l=[]
    # Load the large English NLP model
    nlp = spacy.load("en_core_web_sm")

    # Create a doc object and apply NLP on the text
    doc = nlp(text)

    def split_sentences_with_ai(text):
        # Using the sent_tokenize function from the nltk library
        sentences = sent_tokenize(text)
    
        # Return the list of sentences
        return sentences
    
    questions = comms['intents'][2]['patterns']
    conversations = ["Hi, how are you?", 
                 'my frind is cold',
                 'i like potato',
                 'i went to school two days ago',
                 "Nice day today, isn't it?",
                 "I like to play basketball on weekends."]

    # Convert the data into numerical representation
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(questions + conversations)
    y = np.array([0] * len(questions) + [1] * len(conversations))

    # Train the Naive Bayes classifier
    clf = MultinomialNB().fit(X, y)

    # Test the classifier with new sentences
    new_sentences = split_sentences_with_ai(text)
    X_test = vectorizer.transform(new_sentences)
    predictions = clf.predict(X_test)

    # Print the results
    for i, prediction in enumerate(predictions):
        if prediction == 0:
            l.append(new_sentences[i])
    return l

def txtcllassfie(txxt,json):
    # Define the categories of text inputs
    categories = ['story', 'command', 'qustion', 'conversation', 'facts']
    # Define the training data
    comm=json
    
    training_data=[]
    for i in comm['intents']:
        clas=i['tag']
        for j in i['patterns']:
            txt=j 
            training_data.append((txt,clas))
    # Preprocess the training data and create a list of tuples containing the text and category
    processed_data = [(preprocess(text), category) for text, category in training_data]

    
    # Create a frequency distribution of the words in the training data
    all_words = []
    for words, category in processed_data:
        all_words.extend(words)
    fdist = FreqDist(all_words)
    
    # Create a list of feature sets
    feature_sets = [(extract_features(text,fdist), category) for (text, category) in processed_data]  
      
    # Train the Naive Bayes classifier on the feature sets
    classifier = NaiveBayesClassifier.train(feature_sets)
    
    # Test the AI on a new text input
    processed_text = preprocess(txxt)
    features = extract_features(processed_text,fdist)
    return classifier.classify(features)

def autolrn(sen,tag,lrn):
    addt=input('is the prdiction true and cant have any other tag in other contics\n>> ')
    if addt=='yes'or addt=='yup'or addt=='true'or addt=='TRUE'or addt=='True':
        addto=True
    else:
        addto=False
    if addto:
        for i in intents['intents']:
            if tag ==i['tag']:
                if sen in i['patterns']:
                    pass
                else:
                    i['patterns'].append(sen)
            else :
                pass
        if lrn=='emo':
            with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','w')) as file:
                    json.dump(intents,file,indent=6)
        if lrn=='int':
            with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','w')) as file:
                    json.dump(intents,file,indent=6)
            
    else:
        print('tell me when to add stuf so i can learn them :)')

def bagw(sentence,wrdspkl):
    # separate out words from the input sentence
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(wrdspkl)
    for w in sentence_words:
        for i, word in enumerate(wrdspkl):
            # check whether the word
            # is present in the input as well
            if word == w:
                # as the list of words
                # created earlier.
                bag[i] = 1
    # return a numpy array
    return np.array(bag)

def predict_class(sentence,wordspkl,classespkl,model):
    bow = bagw(sentence,wordspkl)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) 
               if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classespkl[r[0]],
                            'probability': str(r[1])})
        return return_list

def get_emo(emo_list,emo_json):
    probability=float(emo_list[0]['probability'])
    tag = emo_list[0]['intent']
    list_of_emos=emo_json['intents']
    emogot =''
    for i in list_of_emos:
        if i['tag']==tag:
            emogot=tag
            break
    if (probability<.85):
        emogot='nutral'
    return(emogot)

def get_type(comm_list,comm_json):
    tag = comm_list[0]['intent']
    list_of_comms=comm_json['intents']
    typegot =''
    for i in list_of_comms:
        if i['tag']==tag:
            typegot=tag
            break
    #print(comm_list)
    return(typegot)
    
def get_command(cmnd_list,cmnd_json):
    tag = cmnd_list[0]['command']
    list_of_cmnds=cmnd_json['command']
    cmndgot =''
    for i in list_of_cmnds:
        if i['tag']==tag:
            cmndgot=tag
            break
    return(cmndgot)    

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            
              # prints a random response
            random.choice(i['responses'])  
            break
    #autolrn(message,tag,'int')
    return(result)

def dontSearchyt(txt):
    # List of common "commanding" words to be removed
    commanding_words = ["play", "pause", "stop", "resume", "search", "google", "watch", "listen", "open", "go", "find", "show", "load", "start", "view", "playback"]

    ttxt=word_tokenize(txt)

    # Loop through the input words and add them to the filtered list if they are not in the commanding words list
    for word in ttxt:
        if word.lower() in commanding_words and word==ttxt[0]:
            txt.replace(word.lower(),"")
    if txt=='':
        txt=None
    return txt

def OpnYoutubeVid(vidname):

    # Create a YouTube Data API service instance
    youtube = build('youtube', 'v3', developerKey=YOU_API_KEY)

    # Input the search query
   
    if dontSearchyt(vidname):
        query = dontSearchyt(vidname)
    else :
        print("40o04")
    
    
    # Call the YouTube Data API to search for videos
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id,snippet',
        maxResults=1
    ).execute()

    # Extract the video ID and title of the first result
    video_id = search_response['items'][0]['id']['videoId']
    video_title = search_response['items'][0]['snippet']['title']

    # Print the video title and URL
    print(f"Playing video: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    #print(f"Video URL: {video_url}")

    # Open the video URL in the default web browser
    webbrowser.open(video_url)
    
def OpnGoogle(query):
    Dntgogl(query)
    try:
        # Perform Google search
        url= f"https://www.google.com/search?q={query}"
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 OPR/97.0.0.0"}
        page = requests.get(url,headers=headers)
        soup=BeautifulSoup(page.content,"html.parser")
        first_result = soup.find("div", {"class": "yuRUbf"}).a["href"]
        if first_result:
            # Open first search result in default web browser
            webbrowser.open(first_result)

            print(f"Successfully opened")# : {first_result}")
        else:
            print("No search results found.")
    except Exception as e:
        print(f"Error: {e}") 
    return res
    
def Dntgogl(query):
    """
    Removes common command phrases from a search query, but keeps them if needed.

    Args:
        query (str): The search query to process.

    Returns:
        str: The search query with command phrases removed.
    """
    # List of common command phrases
    command_phrases = [
        "get info on",
        "get information on",
        "search for",
        "google it",
        "google ",
        "google",
        "look up",
        "find out about",
        "tell me about",
        "tell me",
        "define",
        "explain",
        "what is",
        "how to",
        "where to",
        "when to",
        "why is",
        "who is",
        # Add more command phrases as needed
    ]

    # Iterate through the command phrases and remove them from the query
    for phrase in command_phrases:
        if phrase in query:
            query = query.replace(phrase, "").strip()

    return query
    
def opnstream(query):
    def extrctNofStreamer(txt):
        nonowrds=["twitch","youtube","stream","open","play","start","steam","sream","on"]
        c=[]
        txtt= word_tokenize(txt)
        for wrd in txtt:
            if wrd in nonowrds:
                txt=txt.replace(wrd, '')
            else:
                c.append(wrd)
        R=""
        for w in c:
            R=R.join(w)
        return R
    
    name=extrctNofStreamer(query)
    
    if ("youtube"in query) or("Youtube"in query) or("YOUTUBE"in query) :
        stream="youtube"
    elif ("twitch"in query) or("Twitch"in query) or("TWITCH"in query):
        stream="twitch"
    else:
        
        stream=qdb().get_question(f"where does {name} stream now?") 
        #print(f"where does {name} stream now?")
    if "youtube"in stream:
        OpnGoogle(f"{name} live on youtube")
    elif "twitch"in stream:
        OpnGoogle(f"{name} live on twitch")
    else:
        print("X_X")
    
def add_dairy(txt,title=None):
    notdairy=["write to diary", "save this story that happened to", "did i tell you what happened today in school"]
    
    for i in notdairy:
        if i in txt:
            txt=txt.replace(i,'')
    diarydb().add_diary_entry(txt,title)

def runn():
    message = input("what is the right answer?")
    mos=sent_tokenize(message)
    kl=[]
    typpropapilty=0
    emopropapilty=0
    ccc=0
    cmndpropapilty=0
    for message in mos:
        typclass=predict_class(message,commwords,commclasses,commmodle)
        emoclss=predict_class(message,emowords,emoclasses,emomodel)
        cmndclss=predict_class(message,cmndwords,cmndclasses,cmndmodle)
        #ints = predict_class(message)
        #typ=txtcllassfie(message, comms)
        typ = get_type(typclass,comms)
        typpropapilty=typpropapilty+float(typclass[0]['probability'])
        emo= get_emo(emoclss,emos)
        emopropapilty=emopropapilty+float(emoclss[0]['probability'])
        kl.append({"type":typ, "type propapilty": typpropapilty/ len(mos),'emotion': f'{emo}', 'emo propapilty': emopropapilty / len(mos)})

        if typ == "qustion":
            print('google it 4HEAD')
            if message[-1]in [" ","?"]:
                message=message.replace(message[-1], '')
            else:
                pass
            if message== '':
                print('invalid input..')
            else:
                answer = qdb().get_question(message)
                if answer:
                    print(answer)
                else:
                    print("answer was not found.")
        elif typ == "command":
            print("right away")
            ccc=+1
            cmnd = get_type(cmndclss,cmnds)
            cmndpropapilty=cmndpropapilty+float(cmndclss[0]['probability'])
            print(f'command: {cmnd}')
            print(f'propapilty: {cmndpropapilty/ccc }')
            if cmnd == "play youtube":
                OpnYoutubeVid(message)
            elif cmnd == "google it":
                OpnGoogle(message)
            elif cmnd=="open stream":
                opnstream(message)
            elif cmnd=="write to diary":
                add_dairy(message)
            else :
                print("^_^")
        else:
            print(typ)
            print(f'propapilty: {typpropapilty}')
    
    #autolrn(message,emo,'emo')
    #res = get_response(predict_class(message,intswords,intsclasses,chatbotints),intents)

runn()

