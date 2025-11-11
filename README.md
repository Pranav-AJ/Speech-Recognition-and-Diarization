# Speech-Recognition-and-Diarization

# 🎙️ AI-Powered Speech Transcription & Diarization App

> ### 🚀 [Click Here to Try the Live Demo!](https://huggingface.co/spaces/pranav3721/my-transcription-app)

This is a web application, built with Python and Gradio, that provides both high-accuracy Automatic Speech Recognition (ASR) and Speaker Diarization.

It allows a user to either record their voice directly through the microphone or upload an existing audio file. The app processes the audio and returns two key pieces of information:
1.  **A full text transcript** (what was said).
2.  **Speaker timestamps** (who spoke and when).

---

## ✨ Features

* **Dual Input:** Supports both live audio recording and file uploads (`.wav`, `.mp3`, etc.).
* **Fast Transcription:** Uses the `sherpa-ncnn` library for efficient ASR.
* **Accurate Diarization:** Uses `sherpa-onnx` with state-of-the-art Pyannote and 3D-Speaker models to identify different speakers.
* **Simple Web UI:** Built with Gradio for a clean, responsive, and easy-to-use interface.

## 💻 Tech Stack

* **Backend:** Python
* **Web Framework:** Gradio
* **ASR Model:** `sherpa-ncnn` (Streaming Zipformer)
* **Diarization Models:** `sherpa-onnx` with:
    * `pyannote/segmentation-3.0`
    * `3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k`
* **Audio Processing:** `librosa`

---

## 🚀 How to Run Locally

If you want to run this project on your own machine, follow these steps.

### 1. Clone the Repository

```bash
# Replace with your GitHub repository URL
git clone [https://github.com/pranav3721/my-transcription-app.git](https://github.com/pranav3721/my-transcription-app.git)
cd my-transcription-app
```

### 2. Create and Activate a Virtual Environment
```bash
# Create the virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


### 4. Download Models

The pre-trained models are too large for this repository and must be downloaded manually.

* **ASR Models (`sherpa-ncnn`):**

  * Download the **Streaming Zipformer (20M)** model files from [this link](https://www.google.com/search?q=https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/streaming-zipformer.html).

  * Place all the model files (`.param`, `.bin`, `tokens.txt`) inside the `sherpa-ncnn-streaming-zipformer-20M-2023-02-17` folder.

* **Diarization Models (`sherpa-onnx`):**

  * **Segmentation:** Download `model.onnx` from [pyannote/segmentation-3.0](https://www.google.com/search?q=https://huggingface.co/pyannote/segmentation-3.0/blob/main/model.onnx) and place it in the `sherpa-onnx-pyannote-segmentation-3.0` folder.

  * **Embedding:** Download the `3dspeaker...onnx` file from [this Hugging Face repo](https://www.google.com/search?q=https://huggingface.co/ResumeMatch/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k/blob/main/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx) and place it in the root folder.
    *(Note: Your `app.py` script must have the correct file paths to these models).*


### 5.Run the App

```bash
python app.py
```
