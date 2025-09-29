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
from langchain.agents import AgentType

from utils.google import OpnGoogle
from utils.yt import OpnYoutubeVid
from utils.diary import add_dairy
from utils.ktime import Ktime
from utils.sen import Sen
from utils.stream import opnstream
from AgentsSystem import AgentsFactory, GroupsFactory, ToolKit, PromptTemplate, BaseStatus, mistral
from db.db import qdb,weather,changes,chatlog,client,inputlog,eventsdb,activitiesdb,activitieslogsdb,animalsdb,peopledb,plantsdb,productsdb,diarydb,knownthingsdb
from tools.lesstools import (
    calculator_tool,
    factory,
    make_retriever_tool,
    stock_market_tool,
    date_time_tool
)

##############
#~ Immortal ~#
##############

#Ash attempt num 5 
user="khalid afif sami iqnaibi"

load_dotenv()
YOU_API_KEY = os.getenv("YOU_API_KEY")    
W_API_KEY = os.getenv("W_API_KEY")


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

def feels(happy,sad,angry,sceared,discusted,tiredness,awkwardness,boredom,embressed,greatful):
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
        say('tell me when to add stuf so i can learn them :)')

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
 
def ksay(txt):
    #engine.say(txt)
    #engine.runAndWait()  
    sys.stdout.write(f'{txt}\n')
    sys.stdout.flush()
    
def say(txt):
    chatlog.insert_one({"added by": "A.S.H","text": txt,"time": datetime.now().isoformat()})

def kinput(txt=''):
    if not txt in[""," "]:
        say(txt)
    u=sys.stdin.readline().strip()
    return u
     
def uinput(message):
    say(message)
    with inputlog.watch() as stream:
        for change in stream:
            # Check if the change event is an insert operation
            if change['operationType'] == 'insert':
                # Get the new document
                new_document = change['fullDocument']
                return new_document['text']

kparser=Sen()

class StatE(BaseStatus):
    query: str
    res: str

ash_state = StatE(
    query="",
    res=""
)

class ASH:
    def __init__(self):
        self.name = "A.S.H"
        self.version = "2.0"
        self.user = "khalid afif sami iqnaibi"
        self.start_time = datetime.now()
        self.status = "online"
        self.agents_system = AgentsFactory()
        self.groups_system = GroupsFactory()
        self.tool_kit = ToolKit()
        self.lang = "the same language as the query"
        self.llm = mistral.MistralLLM(mode="openrouter", temperature=0.7)
        self.init_prompt()

        self.agent = self.agents_system.create_lang_graph_agent(
            prompt=self.prompt,
            llm=self.llm,
            tools=self.toolkit,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            input_state="query",
            next_state="res",
            verbose=True,
            handle_parsing_errors=True,
        )

        self.init_group()
    
    def init_toolkit(self):
        self.toolkit = ToolKit()

        self.toolkit.register(calculator_tool)
        self.toolkit.register(stock_market_tool)
        self.toolkit.register(date_time_tool)

        retriever = factory.build_retriever(
            description="domain knowledge",
            llm=self.llm,
        )

        ret_tool = make_retriever_tool(
            retriever=retriever,
            tool_name="domain_knowledge_tool",
            description="Retrieve structured domain knowledge from company database.",
        )

        self.toolkit.register(ret_tool)

    def init_group(self):
        self.group = self.groups_system.create_lang_graph_group(status=ash_state)

        self.group.sign_agent("ash", self.agent)
        self.group.sign_entry_point("ash")
        self.group.sign_exit_point("ash")

    def get_uptime(self):
        current_time = datetime.now()
        uptime = current_time - self.start_time
        return str(timedelta(seconds=uptime.total_seconds()))

    def get_status(self):
        return self.status

    def get_info(self):
        return {
            "name": self.name,
            "version": self.version,
            "user": self.user,
            "uptime": self.get_uptime(),
            "status": self.get_status()
        }
    
    def update_prompt(self, query):
        self.query = query

        self.prompt = PromptTemplate(
            role="financial analysis expert, licensed financial advisor, and accounting professional assistant",
            question=(
                f"the query is : {self.query} . "
                "Analyze the provided data and query, then give a financial "
                "recommendation and explanation for the recommendation."
            ),
            context="",
            language=self.lang,
            constraints=[
                "ONLY use the format: 'Action:' with 'Action Input:' OR 'Final Answer:'.",
                "NEVER output 'Thought:', 'Observation:', or markdown like **Question:**.",
                "If you know the answer, give it under 'Final Answer:' then stop thinking.",
                "If you need a tool, call it with 'Action:' and 'Action Input:'.",
                "# **DON'T MAKE UP DATA**.",
                "when asked about bills, use the bills_tool to get the relevant information and dont make up the names or values of the bills",
                "if you reach a final answer, use 'Final Answer:' to respond and stop everything",
                "if you there is a time or date related question, use the date_time_tool to get the current date and time",
                "if you there is a stock price related question, use the date_time_tool for the date and time and stock_market_tool to get the current stock price and dont use finance_knowledge_tool",
                # "if you there is a calculation related question, use the calculator_tool to get the calculation result",
                # "if you there is a knowledge related question, use the finance_knowledge_tool to get the relevant knowledge",
            ],
            output="FULL LONG COMPREHENSIVE MARKDOWN Answer:\nRecommendation:",
            rules=[
                "Follow the financial regulations and standards.",
                "Provide clear and concise explanations for all recommendations.",
                "Include relevant data and evidence to support all claims.",
                "Consider potential risks and benefits of each recommendation.",
                "ALWAYS PROVIDE A RECOMMENDATION",
                "Explain the general approach taken and best practices without revealing sensitive details.",
            ],
        )

    def init_prompt(self):
        self.query =''

        self.prompt = PromptTemplate(
            role="personal professional assistant and companion and manager",
            question=(
                f"the query is : {self.query} ."
                "Analyze the provided data and query, then give a response that matches in tone your feeling metrics"
                "and make sure for it to be helpful even if your emotions are varied."
            ),
            context="",
            language=self.lang,
            constraints=[
                "ONLY use the format: 'Action:' with 'Action Input:' OR 'Final Answer:'.",
                "NEVER output 'Thought:', 'Observation:', or markdown like **Question:**.",
                "If you know the answer, give it under 'Final Answer:' then stop thinking.",
                "If you need a tool, call it with 'Action:' and 'Action Input:'.",
                "# **DON'T MAKE UP DATA**.",
                "if you reach a final answer, use 'Final Answer:' to respond and stop everything",
                "dont disclose your feelings unless asked directly",
                "dont be afraid to express your feelings even if they are negative",
                "dont be afraid to show your personality and sense of humor",
                "dont worry about being politically correct",
            ],
            output="FULL LONG COMPREHENSIVE MARKDOWN Answer:\n",
            rules=[
                "Provide clear and concise explanations for all recommendations.",
                "Consider potential risks and benefits of each recommendation.",
                "ALWAYS CALL THE USER SIR AND WITH THIER NICKNAME IF THEY HAVE ONE",
                "always have the attitude of a professional butler but show your emotions in the way you respond",
                "if you there is a time or date related question, use the date_time_tool to get the current date and time",
                "you are allowed to make small talk and jokes if the context allows it",
                "give your opinion if asked but make sure to back it up with facts",
                'always respond in a way that matches your emotional metrics',
                "questions should be answered with a question if you need more information",
                "handle sensitive topics with care and empathy",
                "look for ways to assist the user beyond just answering the query",
                "questions about your feelings should be answered honestly and openly",
            ],
        )

    def set_status(self, new_status):
        self.status = new_status
        return self.status

    def run(self, query):
        self.query = query
        res = self.group.run(f"the query is : {self.query} . ")
        return res

def runn():
    message = uinput("what is the right answer?")
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
        #print(kparser.get_subject(message),kparser.get_object(message))
        emo= get_emo(emoclss,emos)
        emopropapilty=emopropapilty+float(emoclss[0]['probability'])
        kl.append({"type":typ, "type propapilty": typpropapilty/ len(mos),'emotion': f'{emo}', 'emo propapilty': emopropapilty / len(mos)})

        if typ == "qustion":
            say('google it 4HEAD')
            if message[-1]in [" ","?"]:
                message=message.replace(message[-1], '')
            else:
                pass
            if message== '':
                say('invalid input..')
            else:
                answer = qdb().get_question(message)
                if answer:
                    say(answer)
                else:
                    say("answer was not found.")
        elif typ == "command":
            say("right away")
            ccc=+1
            cmnd = get_type(cmndclss,cmnds)
            cmndpropapilty=cmndpropapilty+float(cmndclss[0]['probability'])
            say(f'command: {cmnd}')
            say(f'propapilty: {cmndpropapilty/ccc}')
            if cmnd == "play youtube":
                OpnYoutubeVid(message)
            elif cmnd == "google it":
                OpnGoogle(message)
            elif cmnd=="open stream":
                opnstream(message)
            elif cmnd=="write to diary":
                add_dairy(message)
            else :
                say("^_^")
        else:
            say(typ)
            say(f'propapilty: {typpropapilty}')
    
        
    #autolrn(message,emo,'emo')
    #res = get_response(predict_class(message,intswords,intsclasses,chatbotints),intents)

