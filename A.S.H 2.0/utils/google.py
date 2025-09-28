    
def OpnGoogle(query):
    Dntgogl(query)
    try:
        # Perform Google search
        url= f"https://www.google.com/search?q={query}"
        headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36 OPR/97.0.0.0"}
        page = requests.get(url,headers=headers)
        soup=BeautifulSoup(page.content,"html.parser")
        first_result = soup.find("div", {"class": "yuRUbf"}).a["href"]
        if first_result:
            # Open first search result in default web browser
            webbrowser.open(first_result)

            say(f"Successfully opened")# : {first_result}")
        else:
            say("No search results found.")
    except Exception as e:
        say(f"Error: {e}") 
    return res
    
def Dntgogl(query):
    """
    Removes common command phrases from a search query, but keeps them if needed.

    Args:
        query (str): The search query to process.

    Returns:
        str: The search query with command phrases removed.
    """
    # List of common command phrases
    command_phrases = [
        "get info on",
        "get information on",
        "search for",
        "google it",
        "google ",
        "google",
        "look up",
        "find out about",
        "tell me about",
        "tell me",
        "define",
        "explain",
        "what is",
        "how to",
        "where to",
        "when to",
        "why is",
        "who is",
        # Add more command phrases as needed
    ]

    # Iterate through the command phrases and remove them from the query
    for phrase in command_phrases:
        if phrase in query:
            query = query.replace(phrase, "").strip()

    return query
   