#!/usr/bin/python3

import sounddevice
import speech_recognition as sr
from llama_cpp import Llama
import pyttsx4
from threaded_buffered_pipeline import buffered_pipeline
import threading

state = 0
cond = threading.Condition()

def listen():
	global cond
	global state

	recognizer = sr.Recognizer()
	microphone = sr.Microphone()

	while True:
		with cond:
			while state != 0:
				cond.wait()

		with microphone as source:
			print("Adjusting for noise...")
			recognizer.adjust_for_ambient_noise(source)

		ready = False
		while not ready:
			with microphone as source:
				try:
					print("Listening...")
					audio = recognizer.listen(source, timeout=3)
					ready = True
				except sr.WaitTimeoutError:
					pass

		try:
			print("Recognizing speech...")
			text = recognizer.recognize_whisper(audio)
			if text and text != " Thank you.":
				yield text[1:]
				print("I heard \"" + text[1:] + "\"")
				with cond:
					state = 1
					cond.notify()
		except sr.RequestError:
			pass
		except sr.UnknownValueError:
			pass
		
def respond(texts,
	max_tokens=256,
	temperature=0.1,
	top_p=0.5,
	echo=False,
	stop=["\""]):
	llm = Llama(model_path="ggml-vicuna-7b-1.1-q4_1.bin", n_ctx=512, n_batch=126)
	prompt = "Lira is having a conversation with their friend Ned. "
	for text in texts:
		prompt += "Ned says, \"" + text + "\" Lira replies \"" 
		print(prompt)
		stream = llm(
			prompt,
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			echo=echo,
			stop=stop,
			stream=True)
		prompt = ""
		for resp in stream:
			yield resp["choices"][0]["text"]
		yield ""

def speak(texts):
	global cond
	global state
	
	engine = pyttsx4.init()
	
	buffer = ""
	ready = False
	for text in texts:
		if len(text) == 0:
			with cond:
				while state != 1:
					cond.wait()
				if ready:
					engine.runAndWait()
				state = 0
				cond.notify()
			continue

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
			print(buffer[0:pos+1])
			buffer = buffer[pos+1:]
			ready = True
			with cond:
				if state == 1:
					engine.runAndWait()

if __name__ == "__main__":
	buffer_iterable = buffered_pipeline()
	text = buffer_iterable(listen())
	output = buffer_iterable(respond(text))
	speak(output)
