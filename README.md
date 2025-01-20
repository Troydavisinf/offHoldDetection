# offHoldDetection

The following repository contains the script to live transcribe using the faster_whisper AI model and then feed it to an ollama model (llama3.1 in this case) to detect whether or not there is someone on the phone or not. <br />
It also contains sample files in the irrelevant-ollama and old-attempts folder that may (highly doubt it) be helpful to read over.

<br />
All code that is actually improtant though, is in v1.py.
<br />

<br />

Download Ollama from the website
|--------------------|
https://ollama.com
<br />

Install the following:
|--------------------|
pip install ollama
pip install sounddevice
pip install faster_whisper
pip install numpy

<br />
In terms of RAM usage, if this program is to intensive for your machine, there are other ollama models to look into.
<br />
No models are going to be listed in this readMe due to the fact that models are coming out fast, and new and imrpoved models may be out down the line.
<br />
To find a good model for your machine / project, look here: 
https://ollama.com/library?sort=popular
<br />

<br />
Easy Access for Instructions for Further Changes: <br />
"Your job is to take in a transcription of one end of a phone call and detect whether or not someone is on that end of the phone. The transcription can have either silence (represented as spaces), words, or a mixture of both. Remember that it can be concluded that someone is on the phone if they are talking and potentially saying things like "Hello?", "Is there anyone there?", and so on. Also keep in mind they do not have to be saying only introductory phrases as stated before. If there is someone speaking, it can be assumed that someone is on the phone due to the transcription quality. After analyzing the transcription, please only say YES if there is someone on the phone and NO if there is no one on the phone. In other words, your response should only be one word long in length, as it is either a YES or NO. Please also disregard phrases such as "Thank you." and "Thank you for watching." as the transcription model is a little faulty and often misinterprets silence. Here is the following transcript: " **CONCAT TRANSCRIPT HERE**
<br />
<br />

If you have questions, email me!
- troy.davis@infinitaz.com (while working at infinitaz)
- troydaviscs@gmail.com (afterwords)
