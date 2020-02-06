import speech_recognition as sr
import os
import time
import playsound
from gtts import gTTS

def speak(text):
    tts = gTTS(text=text, lang="en")
    filename = "voice.mp3"
    tts.save(filename)
    playsound.playsound(filename)

r = sr.Recognizer()

running = True

while running:
    with sr.Microphone() as source:
        print("Say something")
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))

            if "stop" in text:
                running = False;
                break;

            speak(text)


        except sr.UnknownValueError:
            print("sorry couldn't recognize that")
        except sr.RequestError as e:
            print(e)
            print("sorry couldn't recognize that")
    