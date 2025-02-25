import pyttsx3
import datetime
import speech_recognition as sr
import subprocess
import pywhatkit
import webbrowser


engine=pyttsx3.init()
voice=engine.getProperty('voices')
print(voice)
print(voice[1].id)
engine.setProperty('voice',voice[1].id)
recognizer=sr.Recognizer()

def command():
    with sr.Microphone() as source:
        print("Clearing the background noices, Please wait...")
        recognizer.adjust_for_ambient_noise(source,duration=2)
        print("Ask anything")
        recordaudio=recognizer.listen(source)
        cmd=recognizer.recognize_google(recordaudio,language="en-Us")
        print(cmd)

        if "time" in cmd:
            time=datetime.datetime.now().strftime('%I:%M %p')
            print(time)
            engine.say(time)
            engine.runAndWait()
        pass
command()
