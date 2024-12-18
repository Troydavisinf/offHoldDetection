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
                print("... (silence detected)")
                #continue ### CHECK THIS LATER WITH MICROPHONE TO SEE IF IT TRANSCRIBES BLANKS OR NOTHING AT ALL

            # Transcribe audio
            segments, _ = model.transcribe(audio_chunk, language="en")

            # Print each word on a new line
            for segment in segments:
                words = segment.text.split()

                transcription_buffer.extend(words)

                for word in words:
                    print(word) # Each word print out

                if time.time()-last_flush_time >= 3:
                    threeText = " ".join(transcription_buffer)
                    print(threeText)

                    # Reset buffer & timer
                    transcription_buffer = []
                    last_flush_time = time.time()


                    response = ollama.chat(
                        model='llama3.2',
                        messages=[{
                            'role': 'user',
                            'content': """The following is a live transcription of a call. Please indicate whether """ +
                            """or not there is someone on the phone. It would be evident that someone is now on the """ +
                            """phone if they are speaking and/or potentially saying things like hello?, hi, is there """ +
                            """anyone there? If there is silence (or nothing passed in as the transcript past the colon), """ +
                            """it is can be concluded that no one is on the phone so you should respond with NO. Please """ +
                            """respond with only either YES if someone is on the phone or NO if no one is on the phone """ +
                            """given the following transcription over 3 seconds (only one side of the call meaning """ +
                            """there will not be two people on the phone):""" +threeText
                        }]
                    )

                    print(response['message']['content'])

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    real_time_transcription()
