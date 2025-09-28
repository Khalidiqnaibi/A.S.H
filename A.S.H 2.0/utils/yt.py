
def dontSearchyt(txt):
    # List of common "commanding" words to be removed
    commanding_words = ["play", "pause", "stop", "resume", "search", "google", "watch", "listen", "open", "go", "find", "show", "load", "start", "view", "playback"]

    ttxt=word_tokenize(txt)

    # Loop through the input words and add them to the filtered list if they are not in the commanding words list
    for word in ttxt:
        if word.lower() in commanding_words and word==ttxt[0]:
            txt.replace(word.lower(),"")
    if txt=='':
        txt=None
    return txt

def OpnYoutubeVid(vidname):

    # Create a YouTube Data API service instance
    youtube = build('youtube', 'v3', developerKey=YOU_API_KEY)

    # Input the search query
   
    if dontSearchyt(vidname):
        query = dontSearchyt(vidname)
    else :
        say("40o04")
    
    
    # Call the YouTube Data API to search for videos
    search_response = youtube.search().list(
        q=query,
        type='video',
        part='id,snippet',
        maxResults=1
    ).execute()

    # Extract the video ID and title of the first result
    video_id = search_response['items'][0]['id']['videoId']
    video_title = search_response['items'][0]['snippet']['title']

    # Print the video title and URL
    say(f"Playing video: {video_title}")
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    #say(f"Video URL: {video_url}")

    # Open the video URL in the default web browser
    webbrowser.open(video_url)
    