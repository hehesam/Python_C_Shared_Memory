import speech_recognition as sr

def speak():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('Speak : ')
        audio = r.listen(source)


        try:
            text = r.recognize_google(audio)
            print('You said : {}'.format(text))
            print(type(text))

        except:
            print('did not hear it ')

        arr = text.split(" ")
        nrr = []
        for i in arr:
            dd = float(i)
            nrr.append(dd)

        return nrr;


