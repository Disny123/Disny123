import speech_recognition as sr
import os
import pyttsx3
import webbrowser
import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

engine = pyttsx3.init()
recognizer = sr.Recognizer()
model = SentenceTransformer('all-MiniLM-L6-v2')

intents = {
    "open browser":lambda:webbrowser.open('https://www.google.com'),
    "search web":lambda query:
    webbrowser.open(f"https://www.bing.com/search?q={query}"),
    "exit program":lambda:exit()
}
intent_phrases = list(intents.keys())
intent_embeddings = model.encode(intent_phrases)


def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_command():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio).lower()
        except sr.UnknownValueError:
            speak("Sorry, I didn't understand!")
            return ""
        
        
def match_intent(command):
    input_embedding = model.encode([command])
    scores = cosine_similarity(input_embedding,intent_embeddings)[0]
    best_idx = scores.argmax()
    best_score = scores[best_idx]
    if best_score>0.6:
        return intent_phrases[best_idx]
    return "search web"


speak("How can I help you?")
while True:
    command = listen_command()
    if command:
        intent = match_intent(command)
        if intent=="search web":
            intents[intent](command)
            speak(f"Searching Bing for {command}")
        elif intent == "exit program":
            speak("GoodBye!")
            intents[intent]()
        else:
            speak(f"executing:{intent}")
            intents[intent]()

        