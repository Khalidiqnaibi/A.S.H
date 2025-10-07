 
def add_dairy(txt,title=None):
    notdairy=["write to diary", "save this story that happened to", "did i tell you what happened today in school"]
    
    for i in notdairy:
        if i in txt:
            txt=txt.replace(i,'')
    diarydb().add_diary_entry(txt,title)
  