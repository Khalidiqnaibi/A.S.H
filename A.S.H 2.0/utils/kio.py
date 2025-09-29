import pyttsx3
from datetime import datetime

from db.db import chatlog,inputlog

# Initialize the TTS engine
engine = pyttsx3.init()
voice = engine.getProperty('voices')[1]
engine.setProperty('voice', voice.id)
engine.setProperty('rate', 225)  # Speed of speech

def say(txt):
    engine.say(txt)
    engine.runAndWait() 
    print(txt)
    add_log(txt)

def add_log(txt,by="A.S.H"):
    chatlog.insert_one({"added by": by,"text": txt,"time": datetime.now().isoformat()})

def kinput(message,by='user'):
    add_log(message,by=by)
    with inputlog.watch() as stream:
        for change in stream:
            # Check if the change event is an insert operation
            if change['operationType'] == 'insert':
                # Get the new document
                new_document = change['fullDocument']
                return new_document['text']
