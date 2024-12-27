from faster_whisper import WhisperModel
import torch

def transcribe_audio(audio_path, model_size="tiny"):
    # Initialize the model
    model = WhisperModel(
        model_size,
        device='cuda',
        compute_type="int8"  # Reduces memory usage
    )

    # Transcribe with additional options
    segments, info = model.transcribe(
        audio_path,
        language="en",  # Optional: specify language
        beam_size=5,  # Improves accuracy
        best_of=5  # Generate multiple candidates
    )

    # Collect full transcription
    transcription = " ".join(segment.text for segment in segments)

    return transcription

    # return {
    #     "transcription": transcription,
    #     "language": info.language,
    #     "duration": info.duration
    # }


# Example usage
result = transcribe_audio("vocal-spoken-nasa-type_75bpm_C_major.wav")
print(result)