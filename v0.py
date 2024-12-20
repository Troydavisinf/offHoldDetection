import time
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import ollama
from pyexpat.errors import messages


def is_silent(audio, threshold=0.005): # checks if audio is near silent
    energy = np.sqrt(np.mean(audio ** 2))
    return energy < threshold

def real_time_transcription():
    # Load the Whisper model
    model = WhisperModel("tiny", device="cuda", compute_type="float16")
    sample_rate = 16000  # hz
    chunk_duration = 2  # Duration of each recording chunk in seconds
    chunk_size = int(sample_rate * chunk_duration)

    print("Good to go\n")
    transcription_buffer = []
    last_flush_time = time.time()

    try:
        while True:
            # Record audio chunk
            audio_chunk = sd.rec(chunk_size, samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished

            # Convert audio to 1D NumPy array
            audio_chunk = audio_chunk.squeeze()

            # Ignore silent chunks
            if is_silent(audio_chunk):
                print("Silence Detected, will not be passed to AI")
                continue

            # Transcribe audio
            segments, _ = model.transcribe(audio_chunk, language="en")

            # Print each word on a new line
            for segment in segments:
                words = segment.text.split()

                transcription_buffer.extend(words)

                for word in words:
                    print(word) # Each word print out
                    transcription_buffer.append(" ") # Adds spaces after each word or no word, to ensure silence is valued

                if time.time()-last_flush_time >= 4.0:
                    threeText = " ".join(transcription_buffer)
                    # print(threeText)

                    # Reset buffer & timer
                    transcription_buffer = []
                    last_flush_time = time.time()

                    print(threeText)
                    response = ollama.chat(
                        model='llama3.1',
                        messages=[{
                            'role': 'user',
                            'content': """Your job is to take in a transcription of one end of a phone """ +
                                       """call and detect whether or not someone is on that end of the phone. """ +
                                       """The transcription can have either silence (represented as spaces), """ +
                                       """words, or a mixture of both. Remember that it can be concluded that """ +
                                       """someone is on the phone if they are talking and potentially saying """ +
                                       """things like "Hello?", "Is there anyone there?", and so on. Also keep """ +
                                       """in mind they do not have to be saying only introductory phrases as """ +
                                       """stated before. If there is someone speaking, it can be assumed that """ +
                                       """someone is on the phone due to the transcription quality. After """ +
                                       """analyzing the transcription, please only say YES if there is someone """ +
                                       """on the phone and NO if there is no one on the phone. In other words, """ +
                                       """your response should only be one word long in length, as it is either """ +
                                       """a YES or NO. Here is the following transcript:  """ + threeText
                        }]
                    )

                    print(response['message']['content'])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    real_time_transcription()
