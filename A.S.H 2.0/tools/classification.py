import random,json,pickle, spacy,nltk,string,requests,pyttsx3
import sys, os,pymongo,webbrowser
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
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
  
nlp = spacy.load('en_core_web_md')
propapilty=float (0.001)
emotion='nutral'

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
 