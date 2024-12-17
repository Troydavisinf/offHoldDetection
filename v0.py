import time
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import ollama

def is_silent(audio, threshold=0.005): # checks if audio is near silent
    energy = np.sqrt(np.mean(audio ** 2))
    return energy < threshold

def real_time_transcription():
    # Load the Whisper model
    model = WhisperModel("small.en", device="cuda", compute_type="float16")
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
            audio_chunk = audio_chunk.flatten()

            # Ignore silent chunks
            if is_silent(audio_chunk):
                print("... (silence detected)")
                continue

            # Transcribe audio
            segments, _ = model.transcribe(audio_chunk, language="en")

            # Print each word on a new line
            for segment in segments:
                words = segment.text.split()

                transcription_buffer.extend(words)

                for word in words:
                    print(word) # Each word print out

                if time.time()-last_flush_time >= 5:
                    fiveText = " ".join(transcription_buffer)
                    print(fiveText)

                    # Reset buffer & timer
                    transcription_buffer = []
                    last_flush_time = time.time()

                    response = ollama.chat(
                        model='llama3.2',
                        messages=[{
                            'role': 'user',
                            'content': """The following is a live transcription of a call. Please indicate whether """ +
                            """or not there is someone on the other end of the phone. It would be evident that """ +
                            """someone is now on the phone if they are speaking. Please respond with either YES """ +
                            """if someone is on the other end or NO if no one is on the phone given the following """ +
                            """transcription over 5 seconds: """ + fiveText
                        }]
                    )

                    print(response['message']['content'])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    real_time_transcription()
