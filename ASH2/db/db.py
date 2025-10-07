import random
import json
import requests
from bs4 import BeautifulSoup
import pymongo
from datetime import timedelta,datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist


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

class activitiesdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newactivity"})
        del c["_id"]
        self.newAct=c
        
    def addActivitie(self,catagory,value):
        act=self.newAct
        act.update({catagory:value})
        activities.insert_one(act)
        ch={"change":f"added new activity with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getact(self,param,val):
            act=activities.find_one({param: val})
            return act

    def delAct(self,catagory,value):
        activities.find_one_and_delete({catagory:value})
        ch={"change":f"deleted an activitie with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def UpdateOneAct(self,param,val,catagory,newvalue):
        act=self.getact(param, val)
        ch={"change":f"updated activity with the catagory: {catagory} at the value: {newvalue}  for the activity: ({act['name']}),old values are: ({catagory},{act[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        act.update({catagory: newvalue})
        activities.find_one_and_replace({param: val}, act)
    
    def UpdatenewAct(self,catagory,newvalue):
        newAct=self.newAct
        ch={"change":f"updated newAct with the catagory: {catagory} at the value: {newvalue} for : (newAct), old values are: ({catagory},{newAct[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newAct.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newAct"},newAct)

class activitieslogsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newactivities_log"})
        del c["_id"]
        self.newActlog=c
        
    def addActlog(self,personid,date,activitieid):
        log=self.newActlog
        log['person id']=personid
        log['date']=date
        log['activitie id']=activitieid
        actlogs.insert_one(log)
        ch={"change":f"added new actionlog with person id: ({personid}) with activity id: ({activitieid})","time":datetime.now()}
        changes.insert_one(ch)

    def delActlog(self,personid,activityid):
        actlogs.find_one_and_delete({"person id": personid,"activity id": activityid})
        ch={"change":f"deleted an actlog with the person id: {personid} at the activity id: {activityid}","time":datetime.now()}
        changes.insert_one(ch)
    
    def addActLogwithdata(self,catagory,addedData):
        log=self.newActlog
        log.update({catagory:addedData})
        actlogs.insert_one(log)
        ch={"change":f"added new actionlog with the catagory: {catagory} at the value: {addedData}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getactlog(self,personid,activityid):
            log=actlogs.find_one({"person id": personid,"activity id": activityid})
            return log

    def UpdateOneActlog(self,personid,activityid,catagory,newvalue):
        log=self.getactlog(personid,activityid)
        ch={"change":f"updated actionlog with the catagory: {catagory} at the value: {newvalue} for the activtylog: (person id :{personid},activity id:{activityid}),old values are: ({catagory},{log[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        log.update({catagory: newvalue})
        actlogs.find_one_and_replace({"person id": personid,"activity id": activityid}, log)
                
    def UpdatenewActlog(self,catagory,newvalue):
        newActlog=self.newActlog
        ch={"change":f"updated newActlog with the catagory: {catagory} at the value: {newvalue} for : (newActlog), old values are: ({catagory},{newActlog[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newActlog.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newActlog"},newActlog)

class organismsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"neworganism"})
        del c["_id"]
        self.newOrganism=c
        
        
    def addOrganism(self,catagory,value):
        org=self.newOrganism
        org.update({catagory:value})
        organisms.insert_one(org)
        ch={"change":f"added an new organism with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def delOrganism(self,catagory,value):
        organisms.find_one_and_delete({catagory:value})
        ch={"change":f"deleted an organism with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getorganism(self,param,val):
            organism=organisms.find_one({param: val})
            return organism

    def UpdateOneOrganism(self,param,val,catagory,newvalue):
        organism=self.getorganism(param, val)
        ch={"change":f"updated organisms with the catagory: {catagory} at the value: {newvalue} for the organism: ({organism['common name']}), old values are: ({catagory},{organism[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        organism.update({catagory: newvalue})
        organisms.find_one_and_replace({param: val}, per)

    def UpdatenewOrganism(self,catagory,newvalue):
        newOrganism=self.newOrganism
        ch={"change":f"updated newOrganism with the catagory: {catagory} at the value: {newvalue} for : (newOrganism), old values are: ({catagory},{newOrganism[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newOrganism.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newOrganism"},newOrganism)

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
        
class plantsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newplant"})
        del c["_id"]
        self.newPlant=c

    def addplant(self,catagory,value):
        anml=self.newPlant
        if catagory in["name"]:
            catagory="common name"
        else:
            pass
        anml.update({catagory:value})
        plants.insert_one(anml)
        ch={"change":f"added an plants with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getplant(self,param,val):
        organism=organisms.find_one({param: val})
        return organism
    
    def UpdateOneplant(self,param,val,catagory,newvalue):
        plant=self.getplant(param, val)
        ch={"change":f"updated plants with the catagory: {catagory} at the value: {newvalue} for the plant: ({plant['common name']}), old values are: ({catagory},{animal[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        plant.update({catagory: newvalue})
        plants.find_one_and_replace({param: val}, plant)
        
    def delplant(self,catagory,value):
        plants.find_one_and_delete({catagory:value})
        ch={"change":f"deleted a plant with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def UpdatenewPlants(self,catagory,newvalue):
        newplant=self.newPlant
        ch={"change":f"updated newplant with the catagory: {catagory} at the value: {newvalue} for : (newplant), old values are: ({catagory},{newplant[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newplant.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newplant"},newplant)

class knownthingsdb():
    def __init__(self):
        self.TheBest="Khalid"
        
    def newknowns(self,fathercollction):
        if fathercollction == "animals":
            newknown=new.find_one({"new":"newknownAnimal"})
        elif fathercollction == "products":
            newknown=new.find_one({"new":"newknownProduct"})
        elif fathercollction == "plants":
            newknown=new.find_one({"new":"newknownPlants"})
        else:
            newknown=new.find_one({"new":"newknown"})
        
        del newknown["_id"]
        
        return newknown
    
    def getfather(self,fathername,dbname):
        father=(dbname.find_one({"name":fathername}))
        return father 
    
    def delKnown(self,catagory,value):
        known_things.find_one_and_delete({catagory:value})
        ch={"change":f"deleted the known with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def addKnown(self,catagory,value,fathername,fatherdbname):
        k=self.newknowns(fatherdbname)
        k.update({catagory:value})
        k["father name"]=fathername
        known_things.insert_one(k)
        ch={"change":f"added new Known {dbname} with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getknown(self,param,val):
            k=known_things.find_one({param: val})
            return k

    def UpdateOneKnown(self,param,val,catagory,newvalue):
        k=self.getknown(param, val)
        ch={"change":f"updated a known {k['father collction']} with the catagory: {catagory} at the value: {newvalue}  for the known {k['father collction']}: ({k['name']}),old values are: ({catagory},{k[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        k.update({catagory: newvalue})
        known_things.find_one_and_replace({param: val}, k)
    
    def UpdatenewPlants(self,fathercollction,catagory,newvalue):
        newknown=self.newknowns(fathercollction)
        ch={"change":f"updated {newknown} with the catagory: {catagory} at the value: {newvalue} for : (newknown), old values are: ({catagory},{newknown[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newknown.update({catagory: newvalue})
        new.find_one_and_replace({"new": f"{newknown}"},newknown)
        
class productsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newproduct"})
        del c["_id"]
        self.newProduct=c
        
    def addProduct(self,catagory,value):
        prod=self.newProduct
        prod.update({catagory:addedData})
        products.insert_one(prod)
        ch={"change":f"added new product with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getprod(self,param,val):
            prod=products.find_one({param: val})
            return prod

    def delProduct(self,catagory,value):
        products.find_one_and_delete({catagory:value})
        ch={"change":f"deleted a product with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def UpdateOneProd(self,param,val,catagory,newvalue):
        prod=self.getprod(param, val)
        ch={"change":f"updated product with the catagory: {catagory} at the value: {newvalue}  for the activity: ({prod['name']}),old values are: ({catagory},{prod[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        prod.update({catagory: newvalue})
        products.find_one_and_replace({param: val}, prod)
        animals.find_one_and_replace({param: val}, animal)

    def UpdateProd(self,param,val,catagory,newvalue):
        prod=self.getprod(param, val)
        ch={"change":f"updated product with the catagory: {catagory} at the value: {newvalue}  for the activity: ({prod['name']}),old values are: ({catagory},{prod[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        prod.update({catagory: newvalue})
        products.find_one_and_replace({param: val}, prod)
        animals.find_one_and_replace({param: val}, animal)
        new['newProduct'].update({catagory: newvalue})
        with (open('C:/Users/khaaf/Desktop/code/python/newt.json','w')) as file:
                json.dump(new,file,indent=6)
    
    def UpdatenewProduct(self,catagory,newvalue):
        newProduct=self.newProduct
        ch={"change":f"updated newProduct with the catagory: {catagory} at the value: {newvalue} for : (newProduct), old values are: ({catagory},{newProduct[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newProduct.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newProduct"},newProduct)

class eventsdb():
    def __init__(self):
        self.TheBest="Khalid"
        c=new.find_one({"new":"newevent"})
        del c["_id"]
        self.newEvent=c
        
    def addEvent(self,catagory,value):
        ev=self.newEvent
        ev.update({catagory:value})
        events.insert_one(act)
        ch={"change":f"added new activity with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def delEvent(self,catagory,value):
        events.find_one_and_delete({catagory:value})
        ch={"change":f"deleted an event with the catagory: {catagory} at the value: {value}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getevent(self,param,val):
            ev=events.find_one({param: val})
            return ev

    def UpdateOneEvent(self,param,val,catagory,newvalue):
        ev=self.getevent(param, val)
        ch={"change":f"updated events with the catagory: {catagory} at the value: {newvalue}  for the event: ({ev['name']}),old values are: ({catagory},{ev[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        ev.update({catagory: newvalue})
        events.find_one_and_replace({param: val}, ev)
    
    def UpdatenewEvent(self,catagory,newvalue):
        newEvent=self.newEvent
        ch={"change":f"updated newEvent with the catagory: {catagory} at the value: {newvalue} for : (newEvent), old values are: ({catagory},{newEvent[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        newEvent.update({catagory: newvalue})
        new.find_one_and_replace({"new": "newEvent"},newEvent)

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
        say('Question added successfully!')

    def edit_question(self, question, new_answer):
        query = {'question': question}
        answer=self.collection.find_one(query)['answer']
        new_data = {'$set': {'answer': new_answer}}
        ch={"change":f"edited the question {question} With the new answer : {new_answer}, old answer is : {answer}","time":datetime.now()}
        changes.insert_one(ch)
        self.collection.update_one(query, new_data)
        say('Question updated successfully!')

    def delete_question(self, question):
        query = {'question': question}
        answer=self.collection.find_one(query)['answer']
        self.collection.delete_one(query)
        ch={"change":f"deleted the question {question} With the answer : {answer}","time":datetime.now()}
        changes.insert_one(ch)
        say('Question deleted successfully!')

    def get_question(self, question):
        """Get the answer to a question from the database, or search for the answer if not found."""
        query = {'question': question}
        question_data = self.collection.find_one(query)

        if question_data:
            return question_data['answer']
        else:
            say('Question not found in the database. Searching for the answer...')
            answer = self.googlit(question)
            if answer[1]=='result was not found!':
                n=uinput("what is the right answer?")
                if n in ["idk",'i dont know','i do not now','Idk',"IDK","I do not know","I dont know"]:
                    return None
                else:
                    self.add_question(question, n,f"{user}")
                    return n
            else:
                an=uinput(f"is {answer[1]} the correct answer?")
                if an in ["yah",'yes','of course','yup','yah','ya','yee','ye','idk','i dont know','i do not know']:
                    self.add_question(question, answer[1],answer[2])
                    return answer[1]
                else:
                    n=uinput("what is the right answer?")
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

class relationshipdb():
    def __init__(self):
        self.TheBest="Khalid"
        
    def newrelation(self,relationship):
        newr=new.find_one({"relationship":relationship})
        if newr:
            '''in["ownership","ownerships"]:
            newr=new.find_one({"new":"newonership"})
        elif relationship in["siblings","sibling","brothers","sisters","brother","sister"]:
            newr=new.find_one({"new":"newsibling"})
        elif relationship in["roommate","roommates","room mates" "room mate"]:
            newr=new.find_one({"new":"newroommate"})
        elif relationship in["teammate","teammates","team mates","team mate"]:
            newr=new.find_one({"new":"newteammate"})
        elif relationship in["class mates","classmates","classmate","class mate"]:
            newr=new.find_one({"new":"newclassmate"})
        elif relationship in["neighbor","neighbors"]:
            newr=new.find_one({"new":"newneighbor"})
        elif relationship in["parent_son","parent-son","parent son","parent","son"]:
            newr=new.find_one({"new":"newparentson"})'''
            pass
        else:
            newr={
                "new": f"new{relationship}",
                "relationship": relationship,
                "names": [],
                "ids": [],
                "date the relationship started": "0010-01-01T01:01:01",
                "date the relationship ended": "0010-01-01T01:01:01",
                "relationship discripion": "",
                "other info": []
            }
        if "_id"in newr.keys():
            del newr["_id"]
        else: 
            pass
        return newr
    
    def addperntson(self,parentname,parentid,sonname,sonid):
        newr=self.newrelation("parent_son")
        newr.update({"parent name":parentname,"son name":sonname,"parent id":parentid,"son id":sonid})
        relationships.insert_one(newr)
        ch={"change":f"added a new parent-son relationship with the parent name: {parentname} and the son name:{sonname}","time":datetime.now()}
        changes.insert_one(ch)
    
    def addownership(self,ownertname,ownertid,ownedname,ownedid,ownedtype,ownertype):
        newr=self.newrelation("ownership")
        newr.update({"owner name":ownertname,"owned name":ownedname,"owner id":ownertid,"owned id":ownedid,"owned collction name":ownedtype,"owner collction name":ownertype})
        relationships.insert_one(newr)
        ch={"change":f"added a new ownership relationship with the owner name: {ownertname} and the owned name:{ownedname}","time":datetime.now()}
        changes.insert_one(ch)
    
    def addRelationship(self,relationship,firstPname,secondPname,firstid,secondid):
        newr=self.newrelation(relationship)
        for j in [parentname,sonname]:
            newr["names"].append(j)
        for l in [parentid,sonid]:
            newr["ids"].append(l)
        relationships.insert_one(newr)
        ch={"change":f"added a new {relationship} relationship with the name: {firstPname} and the name:{secondPname}","time":datetime.now()}
        changes.insert_one(ch)
        
    def delrelationship(self,relationship,firstid,secondid):
        if relationship in["ownership","ownerships"]:
            relationships.find_one_and_delete({"owner id": firstid,"owend id":secondid})
        elif relationship in["parent_son","parent-son","parent son","parent","son"]:
            relationships.find_one_and_delete({"parent id": firstid,"son id":secondid})
        else:
            relationships.find_one_and_delete({"ids": {'$in':[firstid,secondid]}})
        ch={"change":f"deleted a {relationship} relationship between the person id: {firstid} and the person id: {secondid}","time":datetime.now()}
        changes.insert_one(ch)
    
    def getrelationship(self,firstid,seconedid):
            if relationship in["ownership","ownerships"]:
                re=relationships.find_one({"owner id": firstid,"owend id":secondid})
            elif relationship in["parent_son","parent-son","parent son","parent","son"]:
                re=relationships.find_one({"parent id": firstid,"son id":secondid})
            else:
                re=relationships.find_one({"ids": {'$in':[firstid,secondid]}})    
            return re

    def UpdateOnerelationship(self,firstid,seconedid,catagory,newvalue):
        re=self.getrelationship(firstid, seconedid)
        relationship=re["relationship"]
        ch={"change":f"updated a {relationship} relationship with the catagory: {catagory} at the value: {newvalue} for the relationship: (id :{firstid}, id:{seconedid}),old values are: ({catagory},{re[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        re.update({catagory: newvalue})
        if relationship in["ownership","ownerships"]:
            relationships.find_one_and_replace({"owner id": firstid,"owend id":secondid}, re)
        elif relationship in["parent_son","parent-son","parent son","parent","son"]:
            relationships.find_one_and_replace({"parent id": firstid,"son id":secondid}, re)
        else:
            relationships.find_one_and_replace({"ids": {'$in':[firstid,secondid]}}, re)
                
    def UpdatenewRelationship(self,relationship,catagory,newvalue):
        re=self.newrelation(relationship)
        newre=re["new"] 
        ch={"change":f"updated {newre} with the catagory: {catagory} at the value: {newvalue} for : ({newre}), old values are: ({catagory},{re[catagory]})","time":datetime.now()}
        changes.insert_one(ch)
        re.update({catagory: newvalue})
        if new.find_one({"new": newre}):
            new.find_one_and_replace({"new": newre},re) 
        else:
            new.insert_one(re)   
        
    def addperson(self,relationship,pname,pid,seconedid):
        re=relationships.find_one({"relationship":relationship,"ids":{"$in":seconedid}})
        re["ids"].append(pid)
        re["names"].append(pname)
        relationships.find_one_and_replace({"relationship":relationship,"ids":{"$in":seconedid}}, re)
        ch={"change":f"added a new person to the {relationship} relationship with the id: {seconedid} and the id:{pid}","time":datetime.now()}
        changes.insert_one(ch)
  
class diarydb():
    def __init__(self):
        self.KHALID="The Best"

    def generate_diary_title(self,entry):
        # Remove stopwords and tokenize the entry
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(entry.lower())
        words = [word for word in words if word.isalnum() and word not in stop_words]

        # Calculate word frequencies
        fdist = FreqDist(words)

        # Get the most common words
        most_common = fdist.most_common(5)  # You can adjust the number of words to consider

        # Create the title from the most common words
        title = ' '.join([word.capitalize() for word, _ in most_common])

        return title

    def add_diary_entry(self,entry,title=None):
        if not title:
            title=self.generate_diary_title(entry)
        entrys={"title":title,"entry":entry,"date":datetime.now().isoformat().replace("T", " ")}
        dairy.insert_one(entrys)

    def diary_entries(self,date):
        start_of_day = datetime.combine(date, datetime.min.time())
        end_of_day = datetime.combine(date, datetime.max.time())

        query = {
            'date': {
                '$gte': start_of_day.isoformat().replace("T", " "),
                '$lte': end_of_day.isoformat().replace("T", " ")
            }
        }
        entries = dairy.find(query)
        return entries

    def get_diary_entries(self,date):
        entries=self.diary_entries(date)
        if entries:
            # Print the retrieved entries
            res=f"Diary entries for {date}:"
            for entry in entries:
                res+=f"\nTitle: {entry['title']}\nContent: {entry['entry']}\nDate: {entry['date']}"
        else:
            res=f"no entries found for {date}"
        return res

