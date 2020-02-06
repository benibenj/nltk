import speech_recognition as sr
import os
import time
import playsound
from gtts import gTTS

r1 = sr.Recognizer()
r2 = sr.Recognizer()

running = True

def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)


def command():
    with sr.Microphone() as source2:
        print("listening")

        audio2 = r2.listen(source2)

        
        text2 = r2.recognize_google(audio2)
        text2 = text2.lower()
        
        if "stop" in text2:
            running = False

        print(text2)



print("Starting...")
while running:
    with sr.Microphone() as source:

        audio = r1.listen(source)

        try:
            text = r1.recognize_google(audio)
            text = text.lower()

            if "stop" in text:
                running = False;
                break;

            if "jarvis" in text:
                command()

        except sr.UnknownValueError:
            print("...")
        except sr.RequestError as e:
            print(e)
    