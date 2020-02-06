import speech_recognition as sr

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
        except sr.UnknownValueError:
            print("sorry couldn't recognize that")
        except sr.RequestError as e:
            print(e)
            print("sorry couldn't recognize that")
    