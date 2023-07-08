#!/usr/bin/python3

import sounddevice
import speech_recognition as sr
from llama_cpp import Llama
import pyttsx4
from threaded_buffered_pipeline import buffered_pipeline

# create recognizer and mic instances
recognizer = sr.Recognizer()
microphone = sr.Microphone()
llm = Llama(model_path="ggml-vicuna-7b-1.1-q4_1.bin", n_ctx=512, n_batch=126)
engine = pyttsx4.init()

def listen():
	with microphone as source:
		print("Adjusting for noise...")
		recognizer.adjust_for_ambient_noise(source)
		print("Listening...")
		audio = recognizer.listen(source)

	text = ""
	err = None
	try:
		print("Recognizing speech...")
		text = recognizer.recognize_whisper(audio)
		print("I heard \"" + text + "\"")
	except sr.RequestError:
		err = "API unavailable"
	except sr.UnknownValueError:
		err = "Unable to recognize speech"
		print("Adjusting for noise...")
		with microphone as source:
			recognizer.adjust_for_ambient_noise(source)
		print("Done")

	return text, err

def respond(prompt,
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["\""]):
    return llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
				stream=True)

buffer = ""
prompt = "Lira is having a conversation with their friend Ned." 
def speak(text):
	global buffer
	global prompt

	buffer += text
	pos = buffer.rfind('.')
	if pos == -1:
		pos = buffer.rfind('!')
	if pos == -1:
		pos = buffer.rfind('?')
	if pos == -1:
		pos = buffer.rfind(',')

	if pos != -1:
		engine.say(buffer[0:pos])
		engine.runAndWait()
		print(buffer[0:pos+1])
		prompt += buffer[0:pos+1]
		buffer = buffer[pos+1:]

ctx = 1024
if __name__ == "__main__":
	while True:
		text, err = listen()
		if text and text != " Thank you.":
			prompt += " Ned says, \"" + text[1:] + "\" Lira replies \""
			print(prompt)
			buffer_iterable = buffered_pipeline()
			output = buffer_iterable(resp["choices"][0]["text"] for resp in respond(prompt))
			for tok in output:
				speak(tok)
			prompt = ""
			#prompt += "\""
			buffer = ""

		if len(prompt) > ctx:
			prompt = prompt[-ctx:]
