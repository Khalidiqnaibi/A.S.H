import json 
from flask import sessions, sessions, Flask, jsonify, render_template, redirect, session,redirect, request, session, url_for
import os 
from dotenv import load_dotenv
from firebase_admin import db
import firebase_admin
from firebase_admin import credentials
from flask_cors import CORS


with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/comm.json','r') )as f:
    comms=json.load(f)

with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','r') )as f:
    emos=json.load(f)
    
def addtg():
    k=True  
    tag=input('input tag \n>> ') 
    patterns=[]
    t=True
    while (k):
        pt=input('input pattern..if you want to stop type "quit" \n>> ')
        if pt == "quit":
            for i in emos['intents']:
                if tag==i['tag']:
                    t=False
                    for j in patterns:
                        i['patterns'].append(j)
                else:
                    pass    
            if t:        
                emos['intents'].append({"tag":tag,"patterns":patterns})
            print ('thanks for helping to improve project ash :)')
            with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','w')) as file:
                json.dump(emos,file,indent=6)
            k= False
        else:
            patterns.append(pt)
            
def addemo(pt,tag=''):
    t=True
    if tag not in ['', ' ']:
        for i in emos['intents']:
            if tag==i['tag']:
                t=False
                i['patterns'].append(pt)
            else:
                pass    
        if t:        
            emos['intents'].append({"tag":tag,"patterns":[pt]})
        print ('thanks for helping to improve project ash :)')
        with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','w')) as file:
            json.dump(emos,file,indent=6)
            
def addpat(tag):
    k=True
    patterns=[]
    t=True
    while (k):
        pt=input('input pattern..if you want to stop type "quit" \n>> ')
        if pt == "quit":
            for i in emos['intents']:
                if tag==i['tag']:
                    t=False
                    for j in patterns:
                        i['patterns'].append(j)
                else:
                    pass    
            if t:        
                emos['intents'].append({"tag":tag,"patterns":patterns})
            print ('thanks for helping to improve project ash :)')
            with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/emo.json','w')) as file:
                json.dump(emos,file,indent=6)
            k= False
        else:
            patterns.append(pt)
   
def addcomm(pt,tag=''):
    t=True
    if tag not in ['', ' ']:
        for i in comms['intents']:
            if tag==i['tag']:
                t=False
                i['patterns'].append(pt)
            else:
                pass    
        if t:        
            comms['intents'].append({"tag":tag,"patterns":[pt]})
        print ('thanks for helping to improve project ash :)')
        with (open('C:/Users/khaaf/Desktop/code/ash_app/src/ash_ai/comm.json','w')) as file:
            json.dump(comms,file,indent=6)
      

'''     
x= input('input "add" to add patterns and thier tag or "addtg" to add emotion or "addpt" to add pattern to an existing emotion \n>> ')
if x=='add':
    k=True
    tag=input('input tag \n>> ')
    inp=input('input pattern..if you want to stop type "quit" \n>> ')  
    while (k):
        if pt == "quit":
            k= False
        else:
            addemo(inp,tag)
elif x=='addpt':
    addpat(input('input the tag that you want to add to  \n>> '))     
elif x=='addtg':
    addtg()    
else:
    print('sorry wrong input :(')
'''
    


load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.secret_key = 'ImmortalPotato'

r'''
# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"C:\Users\khaaf\Desktop\code\A.S.H\key2.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://monydb-f2cdb-default-rtdb.europe-west1.firebasedatabase.app/'
})
'''

@app.route('/')
def inedx():
    return render_template("yuhuh.html")
    
@app.route('/api/<tag>/emo/<inp>')
def adddssemo(tag,inp):
    addemo(inp,tag)
    return render_template("yuhuh.html")


@app.route('/api/<tag>/comm/<pat>')
def adddsscomm(tag,pat):
    addcomm(pat,tag)
    return render_template("yuhuh.html")

if __name__ == '__main__':
    app.run(debug=False)
     
'''
'''