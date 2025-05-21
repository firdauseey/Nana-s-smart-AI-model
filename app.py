import torch
import numpy as np
from PIL import Image
from gtts import gTTS
import gradio as gr
import time
import tempfile
import os
import zipfile
from datetime import datetime
from ultralytics import YOLO

# ✅ Fix for PyTorch 2.6 safe loading
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
    model = YOLO("yolov5s.pt")

WASTE_CLASSES = {
    'bottle': 'recyclable',
    'cup': 'recyclable',
    'plastic': 'recyclable',
    'apple': 'organic',
    'banana': 'organic',
    'orange': 'organic',
    'carrot': 'organic'
}

TRANSLATIONS = {
    'recyclable': {
        'English': "This is recyclable waste. Please recycle it.",
        'Hausa': "Wannan sharar za a iya sake amfani da ita. Da fatan za a sake amfani da ita.",
        'Yoruba': "Eleyi je idoti to le tun lo. Jowo tun lo.",
        'Igbo': "Nke a bu unyi a pụrụ iji mee ihe ọzọ. Biko mee ka o dị ọhụrụ."
    },
    'organic': {
        'English': "This is organic waste. Dispose of it properly.",
        'Hausa': "Wannan sharar tana da halitta. A jefa ta yadda ya kamata.",
        'Yoruba': "Eleyi je idoti adayeba. Jowo fi si ibi to ye.",
        'Igbo': "Nke a bu unyi sitere n'okike. Biko tụfuo ya nke ọma."
    },
    'unknown': {
        'English': "Object detected.",
        'Hausa': "An gano wani abu.",
        'Yoruba': "A ti ri nkan kan.",
        'Igbo': "A hụrụ ihe."
    }
}

EMOJIS = {
    'recyclable': '♻️',
    'organic': '🍌',
    'unknown': '👁️'
}

chat_log = []
os.makedirs("results", exist_ok=True)

def detect_and_narrate(image, language):
    global chat_log
    chat_log = []

    img = np.array(image)
    results = model(img, verbose=False)[0]
    annotated_img = results.plot()

    detections = results.boxes
    speech_lines = []
    width = image.width

    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        label = model.names[cls_id].lower()

        x1 = int(box.xyxy[0][0])
        x2 = int(box.xyxy[0][2])
        position = "ahead"
        if x2 < width / 3:
            position = "on your left"
        elif x1 > 2 * width / 3:
            position = "on your right"

        base = f"{label} detected {position}."
        waste_type = WASTE_CLASSES.get(label, 'unknown')
        extra = TRANSLATIONS[waste_type][language]
        emoji = EMOJIS[waste_type]
        full_message = f"{emoji} {base} {extra}"

        speech_lines.append(extra)
        chat_log.append(full_message)

    full_speech = " ".join(speech_lines) if speech_lines else TRANSLATIONS['unknown'][language]

    audio_path = ""
    if full_speech:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            try:
                gTTS(text=full_speech, lang=language[:2].lower()).save(fp.name)
            except:
                gTTS(text=TRANSLATIONS['unknown']['English'], lang='en').save(fp.name)
            audio_path = fp.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = f"results/annotated_{timestamp}.jpg"
    txt_path = f"results/chatlog_{timestamp}.txt"
    zip_path = f"results/result_{timestamp}.zip"

    Image.fromarray(annotated_img).save(img_path)
    with open(txt_path, "w") as f:
        f.write("\n".join(chat_log))

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(img_path, arcname=os.path.basename(img_path))
        zipf.write(txt_path, arcname=os.path.basename(txt_path))

    return Image.fromarray(annotated_img), audio_path, "\n".join(chat_log), zip_path

def reset_chat():
    global chat_log
    chat_log = []
    return "", "", "", None

app_title = "🧒 Smart Waste & Object Detector by Nana (Age 15)"
app_description = """
### Hi, I'm Nana from Kaduna 👋 and I built this AI app myself 😄
- Upload a photo or use your webcam 📷  
- My AI will detect things and speak them out loud 🗣️  
- It knows if it's waste and what kind ♻️  
- You can use it in English, Hausa, Yoruba, or Igbo 🌍
"""

with gr.Blocks(title=app_title) as demo:
    gr.Markdown(f"## {app_title}")
    gr.Markdown(app_description)

    with gr.Row():
        with gr.Column(scale=1):
            language = gr.Dropdown(["English", "Hausa", "Yoruba", "Igbo"], label="🌍 Choose a Language", value="English")
            image_input = gr.Image(sources=["upload", "webcam"], type="pil", label="📷 Upload or Use Webcam")
            detect_btn = gr.Button("🔍 Detect Objects")
            reset_btn = gr.Button("♻️ Reset Chat")

        with gr.Column(scale=2):
            image_output = gr.Image(type="pil", label="📌 What the AI Sees")
            audio_output = gr.Audio(type="filepath", label="🔊 What the AI Says")
            chat_output = gr.Textbox(label="💬 What the AI Thinks", lines=10)
            file_output = gr.File(label="📁 Download My Results")

    detect_btn.click(fn=detect_and_narrate, inputs=[image_input, language], outputs=[image_output, audio_output, chat_output, file_output])
    reset_btn.click(fn=reset_chat, outputs=[image_output, audio_output, chat_output, file_output])

demo.launch()
