from models import build_model
import torch
import gradio as gr
import os
import re
from kokoro import generate

MODELS_LIST ={
    "v0_19-full-fp32": "kokoro-v0_19.pth",
    "v0_19-half-fp16": "fp16/kokoro-v0_19-half.pth",
}

VOICEPACK_DIR = "voices"  # Ensure this directory exists in your local_model_path

# Available voices 
CHOICES = {
    '🇺🇸 🚺 American Female ⭐': 'af',
    '🇺🇸 🚺 Bella ⭐': 'af_bella',
    '🇺🇸 🚺 Sarah ⭐': 'af_sarah',
    '🇺🇸 🚹 Michael ⭐': 'am_michael',
    '🇺🇸 🚹 nicole': 'af_nicole',
    '🇺🇸 🚹 sky': 'af_sky',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 British Female emma': 'bf_emma',
    '🇬🇧 🚺 British Female isabella': 'bf_isabella',
    '🇬🇧 🚹 British Male george': 'bm_george',
    '🇬🇧 🚹 British Male lewis': 'bm_lewis',
    
}

# Device Selection
device_options = ["auto", "cpu", "cuda"]

# Initialize model and voices (lazy loading)
MODEL_NAME = None
MODEL = None
MODEL_DEVICE = None
VOICES = {}

# Text normalization functions (simplified)
def normalize_text(text):
    text = text.replace("’", "'")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

SAMPLE_RATE = 24000
def load_model_and_voice(selected_device, model_path, voice):
    global MODEL, VOICES, MODELS_LIST, MODEL_NAME, MODEL_DEVICE
    try :
        if selected_device == "auto":
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif selected_device == "cuda":
            if torch.cuda.is_available():
                print("CUDA is available. Using GPU.")
                device = 'cuda'
            else:
                print("CUDA is not available. Using CPU instead.")
                device = 'cpu'
        else:
            device = 'cpu'
    except Exception as e:
        # print(f"Error: {e}")
        print("CUDA Error is not available. Using CPU instead.")
        device = 'cpu'
    
    # Check if we need to reload the model
    should_reload = (
        MODEL is None or
        MODEL_DEVICE != device or 
        MODEL_NAME != model_path
    )

    if should_reload:
        MODEL = build_model(model_path, device)
        MODEL_NAME = model_path
        MODEL_DEVICE = device
        print(f"Loaded model {model_path} on {device}")

    if voice not in VOICES:
        VOICES[voice] = torch.load(os.path.join(VOICEPACK_DIR, f'{voice}.pt'), weights_only=True).to(device)
        print(f'Loaded voice: {voice} on {device}')

    return MODEL, VOICES[voice]
def generate_audio(text, model_name, voice_name, speed, selected_device):
    if not text.strip():
        return (None, "")
    
    # voice = voice_name
    # model_path = model_name
        # Extract just the voice code from the tuple
    voice = voice_name[1] if isinstance(voice_name, tuple) else voice_name
    # Extract just the model path from the tuple
    model_path = model_name[1] if isinstance(model_name, tuple) else model_name
    
    
    model, voice_data = load_model_and_voice(selected_device, model_path, voice)
    
    
    audio, out_ps = generate(model, text, voice_data, speed=speed, lang=voice[0])

    return (SAMPLE_RATE, audio), out_ps

# Gradio Interface
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Input Text", placeholder="Enter text here...")
            model_dropdown = gr.Dropdown(list(MODELS_LIST.items()), label="Model", value=("v0_19-full-fp32", "kokoro-v0_19.pth"))
            voice_dropdown = gr.Dropdown(list(CHOICES.items()), label="Voice", value=("🇺🇸 🚺 American Female ⭐", "af"))
            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
            device_dropdown = gr.Dropdown(device_options, label="Device", value="auto")
            generate_button = gr.Button("Generate")
        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")
            text_output = gr.Textbox(label="Output Phonemes")

    generate_button.click(
        generate_audio,
        inputs=[text_input, model_dropdown, voice_dropdown, speed_slider, device_dropdown],
        outputs=[audio_output, text_output]
    )

# Run the app
if __name__ == "__main__":
    app.launch(share=True,debug=True)