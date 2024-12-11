import streamlit as st
import whisper  # For transcribing audio using Whisper
import tempfile

# Page configuration
st.set_page_config(
    page_title="Llama OCR (Audio)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("🎙️ Llama Audio Transcription")

# Add clear button to top right
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🗑️"):
        if 'transcription_result' in st.session_state:
            del st.session_state['transcription_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from audio using Whisper and Llama!</p>',
            unsafe_allow_html=True)
st.markdown("---")

# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'm4a'])

    if uploaded_file is not None:
        if st.button("Transcribe Audio", type="primary"):
            with st.spinner("Processing Audio..."):
                try:
                    # Save uploaded audio to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                        temp_audio.write(uploaded_file.read())
                        temp_audio_path = temp_audio.name

                    # Load Whisper model and transcribe audio
                    model = whisper.load_model("base")  # Use a smaller model for faster transcription
                    transcription = model.transcribe(temp_audio_path)

                    # Save result to session state
                    st.session_state['transcription_result'] = transcription['text']

                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")

# Main content area for results
if 'transcription_result' in st.session_state:
    st.markdown("### Transcription Result")
    st.markdown(st.session_state['transcription_result'])
else:
    st.info("Upload an audio file and click 'Transcribe Audio' to see the results here.")
