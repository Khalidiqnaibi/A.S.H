from langchain.tools import tool
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, FreqDist, sent_tokenize
from nltk.corpus import stopwords
import string
import spacy,json  ,pickle
from keras.models import load_model
import numpy as np
import random


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

def clean_up_sentences(sentence: str) -> list:
    """Tokenize and lemmatize a sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def preprocess(text: str) -> list:
    """Remove punctuation, lowercase, tokenize, and remove stopwords."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def extract_features(text: list, fdist: FreqDist) -> dict:
    """Extract features for classifier."""
    words = set(text)
    features = {}
    for word in fdist.keys():
        features[word] = (word in words)
    return features

@tool
def txtcllassfie(txxt: str, json_data: dict) -> str:
    """Classify text into categories using NaiveBayes."""
    categories = ['story', 'command', 'qustion', 'conversation', 'facts']
    comm = json_data
    training_data = []
    for i in comm['intents']:
        clas = i['tag']
        for j in i['patterns']:
            txt = j 
            training_data.append((txt, clas))
    processed_data = [(preprocess(text), category) for text, category in training_data]
    all_words = []
    for words, category in processed_data:
        all_words.extend(words)
    fdist = FreqDist(all_words)
    from nltk.classify import NaiveBayesClassifier
    feature_sets = [(extract_features(text, fdist), category) for (text, category) in processed_data]  
    classifier = NaiveBayesClassifier.train(feature_sets)
    processed_text = preprocess(txxt)
    features = extract_features(processed_text, fdist)
    return classifier.classify(features)

def bagw(sentence: str, wrdspkl: list) -> np.ndarray:
    """Return bag-of-words vector."""
    sentence_words = clean_up_sentences(sentence)
    bag = [0]*len(wrdspkl)
    for w in sentence_words:
        for i, word in enumerate(wrdspkl):
            if word == w:
                bag[i] = 1
    return np.array(bag)

@tool
def predict_class(sentence: str, wordspkl: list, classespkl: list, model) -> list:
    """Predict intent class for a sentence."""
    bow = bagw(sentence, wordspkl)
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.3
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classespkl[r[0]], 'probability': str(r[1])})
    return return_list

@tool
def get_type(comm_list: list, comm_json: dict) -> str:
    """Get type tag from prediction list."""
    tag = comm_list[0]['intent']
    list_of_comms = comm_json['intents']
    typegot = ''
    for i in list_of_comms:
        if i['tag'] == tag:
            typegot = tag
            break
    return typegot
