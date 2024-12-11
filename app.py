import assemblyai as aai
aai.settings.api_key = "00c05f5442794568a81630c6d00110b3"

transcriber = aai.Transcriber()

transcript = transcriber.transcribe("vocal-spoken-nasa-type_75bpm_C_major.wav")

print(transcript.text)