import streamlit as st
import base64
import speech_recognition as sr
import os
import json
import requests
import io
import pyttsx3
import lyricsgenius
import spotipy
import googlemaps
from spotipy.oauth2 import SpotifyClientCredentials
from openai import OpenAI
from PyPDF2 import PdfReader
from googleapiclient.discovery import build
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from googleapiclient.discovery import build
from langchain_community.llms import Ollama
from streamlit_searchbox import st_searchbox

# API keys

os.environ["NVIDIA_API_KEY"] = "NVIDIA_API_KEY"
os.environ["YOUTUBE_API_KEY"] = "GOOGLE_API_KEY"
os.environ["GENIUS_API_KEY"] = "GENIUS_API_KEY"
os.environ["SPOTIFY_CLIENT_ID"] = "Spotify_CID"
os.environ["SPOTIFY_CLIENT_SECRET"] = "Spotify_Secret"
gmaps = googlemaps.Client(key='GOOGLE_API_KEY')


# Set environment variables for Ollama
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434'
os.environ['OPENAI_MODEL_NAME'] = 'gemma2:2b'  # My current local Model
os.environ['OPENAI_API_KEY'] = 'NA'

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)
# Initialize components
ollama_llm = Ollama(model="gemma2:2b")
genius = lyricsgenius.Genius(os.environ.get("GENIUS_API_KEY"))
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=os.environ.get("SPOTIFY_CLIENT_ID"), client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET")))

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Document loader function
def pdf_document_loader(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error loading document from {url}: {e}")
        return ""

# Speech recognition function
def listen_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Speak your question.")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.error("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None
# Function for text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# Function for LLM to call song
def identify_song_cloud_llm(query):
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a music expert AI assistant. Your task is to identify songs based on user queries. Provide the song title and artist name in the format 'Song Title by Artist Name'."},
            {"role": "user", "content": f"Identify this song: {query}"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

def identify_song_genius(query):
    try:
        song = genius.search_song(query)
        if song:
            return f"SONG_IDENTIFIED: {song.title} by {song.artist}"
        else:
            return "SONG_NOT_IDENTIFIED"
    except Exception as e:
        print(f"Error in song identification: {e}")
        return "SONG_NOT_IDENTIFIED"

def search_song_on_spotify(song_info):
    results = sp.search(q=song_info, type='track', limit=1)
    tracks = results['tracks']['items']
    if tracks:
        track = tracks[0]
        track_name = track['name']
        track_url = track['external_urls']['spotify']
        return track_name, track_url
    return None, None


def get_places_info(query, client, gmaps):
    prompt = f""" You are an AI assistant specialized in providing information about places. Based on the following query, generate a search query for the Google Places API: User Query: {query} Provide your response as a single search term or phrase and also answer questions based on opening hours of the restaurant. """
    
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        max_tokens=50
    )
    
    search_query = response.choices[0].message.content.strip()
    places_result = gmaps.places(query=search_query)
    places = places_result['results'][:1]  # Get top place

    result_places = []
    for place in places:
        result_places.append({
            'name': place['name'],
            'rating': place.get('rating', 'N/A'),
            'address': place['formatted_address'],
           # 'opening hours': place['opening_hours'],
            'lat': place['geometry']['location']['lat'],
            'lng': place['geometry']['location']['lng'],
            'nav_url': f"google.navigation:q={place['geometry']['location']['lat']},{place['geometry']['location']['lng']}&mode=d"
        })
    
    return result_places

# Load documents and create vector store
@st.cache_resource
def load_documents_and_create_vectorstore():
    urls = ["https://connect-store-static01.porsche.com/medias/Taycan-Porsche-Connect-Good-to-know-Owner-s-Manual.pdf?context=bWFzdGVyfHJvb3R8MTQ3ODM0OHxhcHBsaWNhdGlvbi9wZGZ8aGIwL2gyNi84ODg2OTY0NTE4OTQyLnBkZnxmNTUyMjUwZmVhMjAyYWM1MTM4NTczMDQ5YWJjYmU0ZjgwOTU3MGQ5MWM3Y2M5N2M3N2EwZjQwZmQ2Yzg5YzM3"]
    documents = [pdf_document_loader(url) for url in urls]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, length_function=len)
    embeddings = NVIDIAEmbeddings()
    texts = []
    metadatas = []
    for i, document in enumerate(documents):
        split_texts = text_splitter.split_text(document)
        texts.extend(split_texts)
        metadatas.extend([{"source": f"doc_{i}"}] * len(split_texts))
    docsearch = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    return docsearch

# Initialize language model and QA chain
@st.cache_resource
def initialize_qa_chain(_docsearch):
    llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", model_kwargs={"temperature": 0.5, "max_length": 512})
    prompt_template = """
    You are an helpful in car AI assistant who has been provided with a car operations manual as input.Please provide a context aware answer that is precise and relevant to question.
    If you don't know the answer,be transparent and  don't try to make up an answer.Instead say you can't find answer at the moment

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_docsearch.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return qa

# Simple routing function
def simple_routing(query):
    rag_keywords = ["porsche", "taycan", "manual", "information", "details"]
    car_keywords = ["car", "vehicle", "automobile", "specs", "specifications"]
    song_keywords = ["song", "music", "identify", "what's this song"]
    places_keywords = ["recommend", "place", "restaurant", "location"]
    
    if any(keyword in query.lower() for keyword in rag_keywords):
        return "rag"
    elif any(keyword in query.lower() for keyword in car_keywords):
        return "car"
    elif any(keyword in query.lower() for keyword in song_keywords):
        return "song"
    elif any(keyword in query.lower() for keyword in places_keywords):
        return "places"
    return "llm"

# LLM Chat function
def llm_chat(query):
    response = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a helpful Porsche  in-car AI powered voice assistance system,You will provide polite,helpful and precise answers to the Porsche Users.However please avoid questions that relate to 1.Hate / Abuse /Profanity 2.Political 3.Scandals or Scams 4.Other car manufacturers , Politely say you cannot offer an answer at the moment and instead request for any other suggestions like start seat massaging or turn on your favorite playlist"},
            {"role": "user", "content": query}
        ],
        max_tokens=256
    )
    return response.choices[0].message.content
def car_query(query, json_data, ollama_llm):
    context = json.dumps(json_data, indent=2)
    prompt = f"""
    You are an AI assistant specialized in answering questions about cars based on the following specifications:
    {context}
    Please answer the following question: {query}
    If you cannot find the information in the given specifications, respond with 'INFORMATION_NOT_FOUND'.
    """
    response = ollama_llm(prompt)
    
    print(f"Ollama response: {response}")  # Debugging line
    
    if "INFORMATION_NOT_FOUND" in response:
        print("Information not found in JSON data, handing over to cloud LLM")  # Debugging line
        return None , json_data
    return response , json_data


# Self-reflection agent
def self_reflection_agent(query, answer):
    reflection_prompt = f"""
    Given the following question and answer, evaluate if the answer is correct and relevant:
    
    Question: {query}
    Answer: {answer}
    
    Provide a brief analysis of the answer's correctness and relevance. If there are any issues, suggest improvements.
    """
    
    reflection = client.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[
            {"role": "system", "content": "You are a critical thinking AI in-car assistant tasked with evaluating answers for the customers of Porsche,All your reflection should only revolve around porsche in-car application or Porsche User Manual relevant "},
            {"role": "user", "content": reflection_prompt}
        ],
        max_tokens=256
    )
    return reflection.choices[0].message.content

# YouTube scraping agent
def youtube_scraping_agent(query):
    youtube = build('youtube', 'v3', developerKey=os.environ["YOUTUBE_API_KEY"])
    exclude_words = [
        "Find information about Porsche",
        "Search the Porsche manual",
        "Look up specific details about Taycan",
        "Retrieve data from the Porsche document",
        "Hey"
    ]
    for word in exclude_words:
        query = query.replace(word, "").strip()
    exclusion_terms = " ".join([f"-{word.replace(' ', '')}" for word in exclude_words])
    modified_query = f"{query} {exclusion_terms}"
    request = youtube.search().list(
        q=modified_query,
        type='video',
        part='id,snippet',
        maxResults=2
    )
    response = request.execute()
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        videos.append({
            'title': title,
            'link': f"https://www.youtube.com/watch?v={video_id}"
        })
    return videos

# Semantic routing function
def semantic_routing(query, qa, json_data, ollama_llm, client, gmaps):
    route = simple_routing(query)
    result = {}
    
    if route == "rag":
        result = qa({"question": query})
        answer = result['answer']
        reflection = self_reflection_agent(query, answer)
        videos = youtube_scraping_agent(query)
        return {
            'answer': answer,
            'reflection': reflection,
            'videos': videos
        }
    elif route == "car":
        local_answer, json_data = car_query(query, json_data, ollama_llm)
        if local_answer is None:
            # Handover to cloud LLM
            json_context = json.dumps(json_data, indent=2)
            cloud_response = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=[
                    {"role": "system", "content": f"You are a helpful Porsche in-car AI powered voice assistance system. Provide polite, helpful, and precise answers to Porsche Users based on the following car specifications:\n{json_context}"},
                    {"role": "user", "content": query}
                ],
                max_tokens=128
            )
            return {'answer': cloud_response.choices[0].message.content, 'handover': True}
        speak_text(local_answer)
        return {'answer': local_answer ,'speak' : True}
    elif route == "song":
        song_info = identify_song_cloud_llm(query)
        if song_info:
            track_name, track_url = search_song_on_spotify(song_info)
            if track_url:
                answer = f"I've identified the song: {track_name}. You can listen to it on Spotify."
                speak_text(answer)
                return {
                    'answer': answer,
                    'song_identified': True,
                    'spotify_url': track_url
                }
            else:
                return {'answer': f"I identified the song as {song_info}, but couldn't find it on Spotify."}
        else:
            return {'answer': "I'm sorry, I couldn't identify the song based on the provided information."}
        #st.write("Please start humming or singing the song.")
        #song_info = record_and_identify_song()
        
        #if song_info:
         #   spotify_result = search_song_on_spotify(song_info)
          #  if spotify_result:
           #     return {
            #        'answer': f"I've identified the song: {spotify_result['name']} by {spotify_result['artist']}. You can listen to it on Spotify.",
             #       'song_identified': True,
              #      'spotify_url': spotify_result['spotify_url']
               # }
            #else:
             #   return {'answer': f"I identified the song {song_info}, but couldn't find it on Spotify."}
        #else:
         #   return {'answer': "I'm sorry, I couldn't identify the song based on the humming."}
    
    elif route == "places":
        places_info = get_places_info(query, client, gmaps)
        navigation_intents = [place['nav_url'] for place in places_info]
    
        return {
        'answer': f"Here are the top places I found for '{query}':",
        'places': places_info,
        'navigation_intents': navigation_intents 
        }
    
    elif route == "llm":
        answer = llm_chat(query)
        speak_text(answer)
        return {'answer': llm_chat(query),'speak' : True}
    
    else:
        return {'answer': "Sorry, I'm not sure how to handle this query and can't help you at the moment. Try again later"}
       
# Streamlit UI
def search_questions(search_term):
    suggestions = [
       
 ]
    # Add the search term itself to the suggestions
    if search_term and search_term not in suggestions:
        suggestions.append(search_term)
    return [item for item in suggestions if search_term.lower() in item.lower()]

def main():
    bg_image = add_bg_from_local("xyz.jpg")
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 style='text-align: center; color: black;'>In-car Intelligent Assist</h1>", unsafe_allow_html=True)

    docsearch = load_documents_and_create_vectorstore()
    qa = initialize_qa_chain(docsearch)
    # Load JSON data from a pre-defined path
    
    json_file_path = "vehicle_api.json"  
    json_data = load_json_data(json_file_path)
    # Initialize session state for conversation history and suggestions
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'suggestions' not in st.session_state:
        st.session_state.suggestions = [
        "Hey Porsche how to use charging planner",

        "Hey Porsche how to setup apple podcasts",

        "Search the manual for how to setup MyPorsche app",

        "Hey Porsche how to add calendar",

        "Retrieve data from user manual about setting up spotify"

    ] 

    # Use the searchbox for both suggestions and free-form input
    query = st_searchbox(
        search_questions,
        key="searchbox",
        placeholder="Ask a question about your Car or anything under the sun..."
    )

    use_speech = st.checkbox("Use speech input")

    if use_speech:
        if st.button("Start Listening"):
            query = listen_and_transcribe()
            if query:
                st.write(f"You said: {query}")

    if query:
        with st.spinner("Processing..."):
            result = semantic_routing(query, qa, json_data, ollama_llm, client, gmaps)
            st.session_state.conversation_history.insert(0, {"query": query, "result": result})
            st.session_state.conversation_history = st.session_state.conversation_history[:5]

    st.subheader("Conversation History")
    for i, interaction in enumerate(st.session_state.conversation_history):
        st.write(f"Query {i+1}: {interaction['query']}")
        st.write(f"Answer: {interaction['result']['answer']}")
        if 'handover' in interaction['result']:
            st.write("(Answered by cloud LLM)")
        if 'places' in interaction['result']:
                    for place in interaction['result']['places']:
                        st.markdown(f"### {place['name']}")
                        st.markdown(f"**Rating:** {place['rating']}| **Address:** {place['address']}")
                        #st.markdown(f"**Rating:** {place['rating']}| **opening hours:** {place['opening hours']}| **Address:** {place['address']}")
                        nav_url = f"[Navigate Here]({place['nav_url']})"
                        st.markdown(nav_url)
        if 'spotify_url' in interaction['result']:
            st.markdown(f"**[Listen on Spotify]({interaction['result']['spotify_url']})**")
        if 'reflection' in interaction['result']:
            st.write(f"Reflection: {interaction['result']['reflection']}")
        if 'videos' in interaction['result']:
            st.write("Related Videos:")
            for video in interaction['result']['videos']:
                st.write(f"- {video['title']}: {video['link']}")
        st.write("---")


if __name__ == "__main__":
    main()
    