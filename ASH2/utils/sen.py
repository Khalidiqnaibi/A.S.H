import spacy
from nltk.tokenize import word_tokenize, sent_tokenize


nlp = spacy.load('en_core_web_md')

class Sen():
    def __init__(self):#,sentence):
        self.khalid='The best'
        self.potato='YES'
        #self.Subject=self.get_subject(sentence)
        #self.Object=self.get_object(sentence)
        #self.Place=self.get_place(sentence)
        #self.Time=self.get_time(sentence)
    def get_subject(self,sentence):
        doc = nlp(sentence)
        for token in doc:
            if ("subj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    def get_object(self,sentence):
        doc = nlp(sentence)
        for token in doc:
            if ("dobj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
    def get_time(self,sentence):
        time={'time':[],'date':[],'day':[]}
        def time_diffr(sentence):
            doc = nlp(sentence)
            m={'time':[],'date':[],'day':[]}
            tims=['all',]
            day=['today','Today','Yesterday','yesterday'
                 ,'monday','Monday','sunday','Sunday',
                 'tuesday','Tuesday','Friday','Friday','Saturday',
                 'saturday','thursday','Thursday','wednesday',
                 'Wednesday']
            for ent in doc.ents:
                k=False
                # Check if the entity is a time, day, or date
                if ent.label_ == "TIME":
                    m['time'].append(ent.text)
                elif ent.label_ == "DATE":
                    for i in day :
                        if i in ent.text:
                            k=True
                            m['day'].append(ent.text)
                    for i in tims:
                        if i in ent.text:
                            k=True
                            m['time'].append(ent.text)
                    if not k:
                        m['date'].append(ent.text)
                elif ent.label_ == "DAY":
                    m['day'].append(ent.text)
                else:
                    pass
            return m
        def get_tim(sentence):
            s=[]
            timeph=['tonight','the evening',"evening",
                    'when i was at school','this week',
                    'next week']
            popo=True
            for i in timeph:
                if i in sentence:
                    for j in s:
                        if ( i in j):
                            popo=False
                    if popo:
                        s.append(i)
                else:
                    pass
            return s
        potato=time_diffr(sentence)
        tm= get_tim(sentence)
        if potato:
            for i in potato['time']:
                time['time'].append(i)
            for j in potato['day']:
                time['day'].append(j)
            for c in potato['date']:
                time['date'].append(c) 
        for k in tm:
            time['time'].append(k)
        return time
    def get_place(self,sentence):
        s=[]
        # Tokenize the sentence into words
        words = word_tokenize(sentence)
        doc=nlp(sentence)
        def pos_tag(wrd) :
            k=[]
            c=0
            dc = doc
            for i in dc:
                if c == wrd:
                    k.append([i.text,i.pos_])
                c=c+1
            return k
        for wrd in range(len(words)):
            if (pos_tag(wrd)[0][1] == "ADP"and words[wrd]not in ['for','with','by','of']or (wrd+1<len(words)and words[wrd] in ['by']and words[wrd+1] in ['the'])):
                plc=''
                if ((wrd+1)<len (words)):
                    i=wrd+1
                    while (pos_tag(i)[0][1] in ["DET","ADJ","PRON","CCONJ",'PROPN',"ADP","NOUN"]and (words[i]not in ['for','with','last','of','by','evenings','evening']or (words[i] in ['by']and words[i+1] in ['the']))):
                        if (pos_tag(i)[0][1]in["DET","ADJ","PRON","CCONJ","ADP"])and((i+1)<len (words))and(pos_tag(i+1)[0][1] not in ["DET",'PROPN',"PRON","ADJ","CCONJ","PRP$","NOUN"]or words[i+1] in ['for','with','last','of','by','evenings','evening']):
                            break
                        elif (pos_tag(i)[0][1]in["DET","ADJ","PRON","CCONJ","ADP"])and((i+1)==len (words)):
                            break
                        else:
                            plc = plc + words[i]+" "
                        if ((i+1)<len (words)):
                            i=i+1
                        else:
                            break
                    if (plc == ''):    
                        pass
                    else:
                        if ((len(s)>0)and(plc in s[-1])):
                            pass
                        else:
                            s.append(plc)
                else:
                    pass
            elif ((wrd>0)and (wrd+1)<len (words)and pos_tag(wrd)[0][1] == 'VERB'and pos_tag(wrd+1)[0][1]in["ADJ","PRON","DET"]and words[wrd+1]not in ['a','an']and pos_tag(wrd-1)[0][1]not in ['PART'])or (words[wrd] in ['visited']):
                plc=''
                if ((wrd+1)<len (words)):
                    i=wrd+1
                    while (pos_tag(i)[0][1] in ["DET","ADJ","PRON",'PROPN',"CCONJ","ADP","NOUN"]and (words[i]not in ['for','with','last','of','by','evenings','evening']or (words[i] in ['by']and words[i+1] in ['the']))):
                        if (pos_tag(i)[0][1]in["DET","ADJ","PRON","CCONJ","ADP"])and((i+1)<len (words))and(pos_tag(i+1)[0][1] not in ["DET",'PROPN',"PRON","ADJ","CCONJ","PRP$","NOUN"]or words[i] in ['for','with','last','of','by','evenings','evening'] ):
                            break
                        elif (pos_tag(i)[0][1]in["DET","ADJ","PRON","CCONJ","ADP"])and(i+1)==len (words):
                            break
                        else:
                            plc = plc + words[i]+" "
                        if ((i+1)<len (words)):
                            i=i+1
                        else:
                            break
                    if (plc == ''):    
                        pass
                    else:
                        if ((len(s)>0)and(plc in s[-1])):
                            pass
                        else:
                            s.append(plc)
                else:
                    pass
        return s
   