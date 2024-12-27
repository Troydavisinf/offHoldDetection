# offHoldDetection

The following repository contains the script to live transcribe using the faster_whisper AI model and then feed it to an ollama model (llama3.1 in this case) to detect whether or not there is someone on the phone or not.

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
Easy Access for Instructions for Further Changes: <br />
"Your job is to take in a transcription of one end of a phone call and detect whether or not someone is on that end of the phone. The transcription can have either silence (represented as spaces), words, or a mixture of both. Remember that it can be concluded that someone is on the phone if they are talking and potentially saying things like "Hello?", "Is there anyone there?", and so on. Also keep in mind they do not have to be saying only introductory phrases as stated before. If there is someone speaking, it can be assumed that someone is on the phone due to the transcription quality. After analyzing the transcription, please only say YES if there is someone on the phone and NO if there is no one on the phone. In other words, your response should only be one word long in length, as it is either a YES or NO. Please also disregard phrases such as "Thank you." and "Thank you for watching." as the transcription model is a little faulty and often misinterprets silence. Here is the following transcript: " **CONCAT TRANSCRIPT HERE**
