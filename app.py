import gradio as gr
import sherpa_ncnn
import sherpa_onnx
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import wave
import os
import time

# --- 1. ASR (sherpa-ncnn) Code ---

def init_asr_recognizer():
    """Loads the ASR model. No Streamlit code here."""
    print("Loading ASR model (sherpa-ncnn)...")
    recognizer = sherpa_ncnn.Recognizer(
        tokens="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/tokens.txt",
        encoder_param="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/joiner_jit_trace-pnnx.ncnn.param",  # <-- This line is now correct
        joiner_bin="sherpa-ncnn-streaming-zipformer-20M-2023-02-17/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
    )
    print("ASR model loaded.")
    return recognizer

def run_asr_transcription(recognizer, samples_float32, sample_rate):
    """
    Runs ASR on the provided audio samples.
    This version uses the older API (from your original script)
    that matches your sherpa-ncnn version.
    """
    
    # Feed the waveform to the recognizer
    recognizer.accept_waveform(sample_rate, samples_float32)

    # Add tail padding
    tail_paddings = np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    recognizer.accept_waveform(sample_rate, tail_paddings)

    # Tell the recognizer we are done
    recognizer.input_finished()
    
    # Get the text
    return recognizer.text


# --- 2. Diarization (sherpa-onnx) Code ---

def init_diarizer(num_speakers: int = -1, cluster_threshold: float = 0.5):
    """Loads the Diarization model."""
    print("Loading Diarization model (sherpa-onnx)...")
    segmentation_model = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
    
    embedding_extractor_model = (
        "3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx"
    )

    # Check if files exist
    if not Path(segmentation_model).is_file():
        print(f"Diarization file not found: {segmentation_model}")
        return None
    if not Path(embedding_extractor_model).is_file():
        print(f"Diarization file not found: {embedding_extractor_model}")
        return None

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers, threshold=cluster_threshold
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        print("Diarization config is invalid. Check model paths.")
        return None

    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    print("Diarization model loaded.")
    return sd

def resample_audio_librosa(audio, sample_rate, target_sample_rate):
    """Resample audio to target sample rate using librosa."""
    if sample_rate != target_sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
    return audio, target_sample_rate

def run_diarization(sd, audio_samples, sample_rate):
    """
    Runs diarization on the provided audio samples.
    """
    if sd is None:
        return ["Diarization model not loaded."]
        
    target_sample_rate = sd.sample_rate
    
    # Resample audio to match the diarizer's expected sample rate
    audio_resampled, _ = resample_audio_librosa(audio_samples, sample_rate, target_sample_rate)

    # Process
    try:
        result = sd.process(audio_resampled).sort_by_start_time()
    except Exception as e:
        print(f"Error during diarization processing: {e}")
        return []

    # Format output
    output_lines = []
    for r in result:
        line = f"{r.start:.3f} -- {r.end:.3f} speaker_{r.speaker:02}"
        output_lines.append(line)
    
    return output_lines


# --- 3. Load Diarizer Globally ---
diarizer = init_diarizer()

if diarizer is None:
    raise RuntimeError("Failed to load Diarization model. Check file paths and error messages.")

DIAR_SAMPLE_RATE = diarizer.sample_rate


# --- 4. The Main Gradio Processing Function ---

def process_audio(audio_filepath):
    """
    This function is called by Gradio when the user clicks 'Submit'.
    It takes the filepath of the recorded audio and returns two strings.
    """
    if audio_filepath is None:
        return "Error: No audio provided.", "Please record or upload audio first."
    
    print(f"Processing audio file: {audio_filepath}")
    
    # Load the audio file (sr=None preserves original sample rate)
    try:
        y, sr = librosa.load(audio_filepath, sr=None)
    except Exception as e:
        return f"Error loading audio: {e}", ""

    # Ensure audio is mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # --- Run ASR ---
    print("Running ASR...")
    start_time = time.time()
    
    # !! Load a FRESH recognizer to avoid state bugs !!
    asr_recognizer = init_asr_recognizer()
    if asr_recognizer is None:
        return "Error: Failed to load ASR model.", ""
        
    ASR_SAMPLE_RATE = asr_recognizer.sample_rate
    
    # 1. Resample for ASR model
    y_asr, _ = resample_audio_librosa(y, sr, ASR_SAMPLE_RATE)
    
    # 2. Call the corrected transcription function
    transcript = run_asr_transcription(asr_recognizer, y_asr, ASR_SAMPLE_RATE)
    print(f"ASR done in {time.time() - start_time:.2f}s")
    
    # --- Run Diarization ---
    print("Running Diarization...")
    start_time = time.time()
    # 1. Diarization function handles its own resampling
    diar_results = run_diarization(diarizer, y, sr)
    # 2. Format results
    if diar_results:
        diar_output = '\n'.join(diar_results)
    else:
        diar_output = "No speakers detected or an error occurred."
    print(f"Diarization done in {time.time() - start_time:.2f}s")

    # Clean up the temp file created by Gradio
    if os.path.exists(audio_filepath):
        os.remove(audio_filepath)

    return transcript, diar_output


# --- 5. Create and Launch the Gradio Interface ---

if __name__ == "__main__":
    print("\n--- Model Sample Rate Info ---")
    print(f"- Diarization Model (sherpa-onnx) expects: {DIAR_SAMPLE_RATE} Hz")
    print("  (ASR Model sample rate will be checked on first run)")
    print("---------------------------------\n")

    # Define the UI
    audio_input = gr.Audio(
        sources=["microphone", "upload"],  # <-- THIS IS THE KEY CHANGE
        type="filepath", 
        label="Record or Upload Your Speech"
    )
    
    output_transcript = gr.Textbox(label="Transcript",lines=6)
    output_diarization = gr.Textbox(label="Speaker Diarization",lines = 6)

    # Create the interface
    iface = gr.Interface(
        fn=process_audio,
        inputs=audio_input,
        outputs=[output_transcript, output_diarization],
        title="🎙️ Audio Transcription and Diarization App",
        description=(
            "Record your speech using the microphone or upload an audio file. " # <-- UPDATED TEXT
            "Click 'Submit' to get the transcript and speaker timestamps."
        )
    )
    
    # Launch the app
    print("Launching Gradio app... Access it at the URL (e.g., http://127.0.0.1:7860)")

    iface.launch(share=True)
