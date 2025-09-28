    
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
        say("X_X")
    