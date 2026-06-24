from datetime import datetime,timedelta
from nltk.tokenize import word_tokenize, sent_tokenize

from utils.sen import Sen

class Ktime():
    def __init__(self):
        self.KHALID="The Best"
    
    def txttoint(self,text):
        """
        Converts number words in text to integers.
        """
        NUMBERS = {
            'zero': 0,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12,
            'thirteen': 13,
            'fourteen': 14,
            'fifteen': 15,
            'sixteen': 16,
            'seventeen': 17,
            'eighteen': 18,
            'nineteen': 19,
            'twenty': 20,
            'thirty': 30,
            'forty': 40,
            'fifty': 50,
            'sixty': 60,
            'seventy': 70,
            'eighty': 80,
            'ninety': 90,
            'hundred': 100,
            'thousand': 1000,
            'million': 1000000,
            'billion': 1000000000,
            'trillion': 1000000000000
        }
        doc = nlp(text)
        nums = []
        nno=0
        for token in doc:
            if token.text.isdigit() or token.text.lower() in NUMBERS:
                nums.append([token.text,doc[nno+1].text])
            else:
                pass
            nno=nno+1
        re=[]
        for i in nums:
            re.append(i[0])
            if not(i[1].isdigit() or i[1] in NUMBERS or i[1] in ['and','-','.']):
                xx=i[1]
                re.append(i[1])
        if (len(re)>0) and(type(re[-1])==int or re[-1].isdigit() or re[-1] in NUMBERS) :
            re.append(xx)
        mm=[]
        mc=[]
        g=1
        gc=0
        result=[]
        cout=-1
        isv=""
        for kl in re:
            cout=cout+1
            if type(kl)==int or kl.isdigit():
                mm.append(int(kl))
            elif kl in NUMBERS and kl not in ['zero','one','two','three','four','five','six','seven','eight', 'nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']:
                mm.append(NUMBERS[kl])
                for k in mm:
                    g=g*k
                mm=[]
                mc.append(g)
                g=1
            elif kl in ['one','two', 'three','four','five','six','seven','eight', 'nine']:
                if (re[cout+1].isdigit())or (re[cout+1] in NUMBERS )or (type(re[cout+1]) == int )or not(cout==len(re)-1):
                    mm.append(NUMBERS[kl])
                else:
                    mc.append(NUMBERS[kl])
            elif kl in ['twenty','thirty','forty','fifty','sixty','seventy','eighty','ninety']:
                mc.append(NUMBERS[kl])
            elif kl in ['zero','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']:
                mm.append(NUMBERS[kl])
                for k in mm:
                    g=g*k
                mm=[]
                mc.append(g)
                g=1
            elif type(kl)==str and cout==len(re)-1:
                isv=kl
            else:
                if mm==[]:
                    pass
                else:
                    for k in mm:
                        g=g*k
                    mm=[]
                    mc.append(g)
                    g=1
                if mc==[]:
                    pass
                else:
                    for l in mc:
                        gc=gc+l
                    result.append(gc)
                    gc=0
                    result.append(kl)
                    mm=[]
                    mc=[]
        if mm==[]:
            pass
        else:
            for k in mm:
                g=g*k
            mm=[]
            mc.append(g)
            g=1
        if mc==[]:
            pass
        else:
            for l in mc:
                gc=gc+l
            result.append(gc)
            mm=[]
            mc=[]
        if not isv == '':
            result.append(isv) 
        return result

    def relative_time(self,time_description,isodate):
        days=0
        mons=0
        wrds=word_tokenize(time_description)
        years=0
        secs=0
        op="pl"

        for wrd in wrds:
            if wrd in ["before","ago"]:
                op="mi"
            elif wrd in ["after"]:
                op="pl"
            else:
                pass

        date=datetime.fromisoformat(isodate)
        re=self.txttoint(time_description)
        for i in range(len(re)):
            if i+1<len(re) and type(re[i])==int:
                if re[i+1] in ["day","days"]:
                    days=re[i]*1
                elif re[i+1] in ["week","weeks"]:
                    days=re[i]*7
                elif re[i+1]in ["months","month"]:
                    mons=re[i]*1
                elif re[i+1] in ["year","years"]:
                    years=re[i]*1
                elif re[i+1] in ["decade","decades"]:
                    years=re[i]*10
                elif re[i+1] in ["century","centurys"]:
                    years=re[i]*100
                elif re[i+1] in ["hour","hours"] :
                    secs=re[i]*3600
                elif re[i+1] in ["minutes","minute"]:
                    secs=re[i]*60
                elif re[i+1]in ["second","seconds"]:
                    secs=re[i]*1
                else:
                    pass
            else:
                pass
        if len(re)<1:
            if "day"in time_description or "days"in time_description :
                days=1
            elif "week"in time_description or "weeks"in time_description:
                days=7
            elif "months"in time_description or "month"in time_description:
                mons=1
            elif "year"in time_description or "years"in time_description:
                years=1
            elif "decade"in time_description or "decades"in time_description:
                years=10
            elif "century"in time_description or "centurys"in time_description:
                years=100
            elif "hour"in time_description or "hours"in time_description:
                secs=3600
            elif "minutes"in time_description or "minute"in time_description:
                secs=60
            elif "second"in time_description or "seconds"in time_description:
                secs=1
            else:
                pass
        else:
            pass
        kha=True
        while kha:
            if (date.month in [0,1,3,5,7,8,10,12]) and (days>30):
                mons=mons+1
                days=days-31
            elif (date.month in [4,6,9,11]) and (days>29):
                mons=mons+1
                days=days-30
            elif (date.month ==2 and days>28) and (date.year%4==0):
                mons=mons+1
                days=days-29
            elif (date.month ==2 and days>27) and not (date.year%4==0):
                mons=mons+1
                days=days-28
            else:
                kha=False

        while mons>11:
            years=years+1
            mons=mons-12

        while secs>86399:
            days=days+1
            secs=secs-86400
            
        if op=="pl":
            result=date
            if date.second+ secs>86399:
                days=days+1
                secs=secs-86400
                result=result.replace(second=secs)
            else:
                result=result.replace(second=date.second+secs)
            
            if date.month in [1,3,5,7,8,10,12] and date.day+days>30:
                mons=mons+1
                days=(date.day+days)-31
                result=result.replace(day=days)
            elif date.month in [4,6,9,11] and date.day+days>29:
                mons=mons+1
                days=(date.day+days)-30
                result=result.replace(day=days)
            elif date.month in [2] and date.day+days>28 and date.year%4==0:
                mons=mons+1
                days=(date.day+days)-29
                result=result.replace(day=days)
            elif date.month in [2] and date.day+days>27 and not date.year%4==0:
                mons=mons+1
                days=(date.day+days)-28
                result=result.replace(day=days)
            else:
                result=result.replace(day=date.day+days)
                
            if (date.month+ mons)>11: 
                years=years+1
                mons=(date.month+ mons)-12
                result= result.replace(month = mons)
            elif mons>0:
                if date.month + mons == 2 and result.day==29 and not date.year + years%4==0:
                    result= result.replace( month = 3, day = 1)
                else:
                    result= result.replace(month = (date.month + mons))
            else:
                pass        
            
            if years>0:
                if result.month == 2 and result.day==29 and not date.year + years%4==0 :
                    result= result.replace(year = (date.year + years), month = 3, day = 1)
                else:
                    result= result.replace(year = (date.year + years))              
        elif op == "mi":
            result=date
            if date.second- secs<0:
                days=days+1
                secs=86400+(date.second-secs)
                result=result.replace(second=secs) 
            else:
                result=result.replace(second=date.second-secs) 
                
            if date.month in [1,3,5,7,8,10,12] and date.day-days<0:
                mons=mons+1
                days=31+(date.day-days)
                result=result.replace(day=days)
            elif date.month in [4,6,9,11] and date.day-days<0:
                mons=mons+1
                days=30+(date.day-days)
                result=result.replace(day=days)
            elif date.month in [2] and date.day-days<0 and date.year%4==0:
                mons=mons+1
                days=29+(date.day-days)
                result=result.replace(day=days)
            elif date.month in [2] and date.day-days<0 and not date.year%4==0:
                mons=mons+1
                days=28+(date.day-days)
                result=result.replace(day=days)
            else:
                result=result.replace(day=date.day-days)
            if date.month- mons<0: 
                years=years+1
                mons=12+(date.month- mons)
                result= result.replace(month = mons)
            elif mons>0:
                if date.month - mons == 2 and result.day==29 and not date.year - years%4==0:
                    result= result.replace( month = 2, day = 28)
                else:
                    result= result.replace(month = (date.month - mons))
            if years>0:
                if result.month== 2 and result.day==29 and not date.year - years%4==0:
                    result= result.replace(year = (date.year - years), month = 2, day = 28)
                else:
                    result= result.replace(year = (date.year - years))
            else:
                pass       
            
                

        else:
            result=date
        return result

    def extract_date(self,txt):
        wrds=word_tokenize(txt)
        da=[]
        for i in range(len(wrds)):
            if not wrds[i] in ['and',"also"]:
                try:
                    date = parse(wrds[i], fuzzy=True)
                    t=datetime.now().time()
                    if i+1<len(wrds)and wrds[i+1]in["am","pm"] or i-1>0 and wrds[i-1]in["at"] :
                        if wrds[i+1] == 'pm':
                            t=date+timedelta(hours=12)
                            t=t.time()
                        else:
                            t=date.time()
                    else:
                        d=date.date()
                except ValueError:
                        pass
            else:
                da.append({"date":d,"time":t})
        da.append({"date":d,"time":t})

        dates=[]

        for dd in da:
            dates.append(datetime.combine(dd["date"],dd["time"]))

        return dates

    def get_whenwas(self,text):
        dates=self.extract_date(text)
        if len(dates)>1: 
            whenwas=[]
            for j in dates:
                res=self.when_was(j)
                k=True
                if res["time"]=="future":
                    whnwas=f'{j} is after'
                elif res['time']=='past':
                    whnwas=f'{j} was'
                elif res["time"]=="now":
                    whnwas=f"{j} is now"
                    whenwas.append({i:whnwas})
                    break
                else:
                    whnwas="4OO0OO4 :("
                    whenwas.append({i:whnwas})
                    break
                for i in res:
                    if not i in["time"]and not res[i]==0 and k:
                        whnwas=whnwas+" "+str(abs(res[i]))+" "+i
                        k=False
                    elif not i in["time"]and not res[i]==0:
                        whnwas=whnwas+" and "+str(abs(res[i]))+" "+i
                    elif res[i]in['past']:
                        whnwas=whnwas+" ago"
                    else:
                        pass
                whenwas.append({i:whnwas})
        elif len(dates)==1:
            res=self.when_was(dates[0])
            k=True
            if res["time"]=="future":
                whenwas=f'{dates[0]} is after'
            elif res['time']=='past':
                whenwas=f'{dates[0]} was'
            elif res["time"]=="now":
                return f"{dates[0]} is now"
            else:
                return "4OO0OO4 :("
            for i in res:
                if not i in["time"]and not res[i]==0 and k:
                    whenwas=whenwas+" "+str(abs(res[i]))+" "+i
                    k=False
                elif not i in["time"]and not res[i]==0:
                    whenwas=whenwas+" and "+str(abs(res[i]))+" "+i
                elif res[i]in['past']:
                    whenwas=whenwas+" ago"
                else:
                    pass
        else:
            whenwas="invalid input.."
        
        return whenwas

    def when_was(self,date):
        result={"years":date.year-datetime.now().year,"monthes":date.month-datetime.now().month,"days":date.day-datetime.now().day,"hours":date.hour-datetime.now().hour,"minutes":date.minute-datetime.now().minute,"seconds":date.second-datetime.now().second,"time":""}
        ww=date-datetime.now()
        if ww.total_seconds() > 0:
            result["time"]="future"
        elif ww.total_seconds() < 0:
            result["time"]="past"
        else:
            result["time"]="now"
        return result

    def get_relative_time(self,time_description,isodate):
        if time_description == "tommorow":
            result= datetime.fromisoformat(isodate)+timedelta(days= datetime.fromisoformat(isodate).day+1)
        elif time_description == "today":
            result= datetime.fromisoformat(isodate)
        elif time_description == "yesterday":
            result= datetime.fromisoformat(isodate)+timedelta(days= datetime.fromisoformat(isodate).day-1)
        else:
            tim=Sen().get_time(time_description)
    
            td=""
            for i in tim:
                for l in tim[i]:
                    td=td+l+" and "
    
            result=self.relative_time(td,isodate) 
        return result
