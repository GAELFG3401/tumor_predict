from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import io
import base64
import os
import gdown
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

# === CONFIG ===
CLASSIFIER_URL = "https://drive.google.com/uc?id=1TQ2_ozS3crjqchAPXNCBuyVs2SvZdf9R"
CLASSIFIER_PATH = "tumor_classifier.h5"

if not os.path.exists(CLASSIFIER_PATH):
    print("Descargando modelo de clasificación...")
    gdown.download(CLASSIFIER_URL, CLASSIFIER_PATH, quiet=False)

app = Flask(__name__)
CORS(app)

resnet_model = load_model(CLASSIFIER_PATH)

def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def base64_to_pil(base64_str):
    return Image.open(io.BytesIO(base64.b64decode(base64_str)))

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    # Clasificación
    img_rgb = Image.open(file).resize((128, 128)).convert('RGB')
    arr_rgb = np.array(img_rgb) / 255.0
    input_rgb = np.expand_dims(arr_rgb, axis=0)

    tumor_prob = resnet_model.predict(input_rgb)[0][0]

    # MRI en escala de grises
    file.seek(0)
    img_gray = Image.open(file).resize((256, 256)).convert('L')
    arr_gray = np.array(img_gray) / 255.0
    mri_base64 = image_to_base64(Image.fromarray((arr_gray * 255).astype(np.uint8)))

    if tumor_prob < 0.0002:
        return jsonify({
            "result": "No hay tumor",
            "probability": float(tumor_prob)
        })

    # Si hay tumor, llamar a la API de segmentación
    file.seek(0)
    response = requests.post("http://localhost:5001/segment", files={"image": file})
    segment_data = response.json()

    # Convertir imágenes base64 a PIL
    mri_img = base64_to_pil(mri_base64)
    mask_img = base64_to_pil(segment_data["mask"])
    overlay_img = base64_to_pil(segment_data["overlay"])

    # Crear PDF en memoria
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setTitle("Reporte de Tumor")

    # Titular
    c.setFont("Helvetica-Bold", 16)
    c.drawString(220, 750, "REPORTE DE TUMOR")

    # Probabilidad
    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Probabilidad de tumor: {tumor_prob:.4f}")

    # Insertar imágenes (escala adecuada)
    mri_reader = ImageReader(mri_img)
    mask_reader = ImageReader(mask_img)
    overlay_reader = ImageReader(overlay_img)

    c.drawString(50, 690, "MRI Original")
    c.drawImage(mri_reader, 50, 500, width=200, height=200)

    c.drawString(300, 690, "Máscara de Tumor")
    c.drawImage(mask_reader, 300, 500, width=200, height=200)

    c.drawString(175, 470, "Overlay con Tumor")
    c.drawImage(overlay_reader, 175, 270, width=200, height=200)

    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="reporte_tumor.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    app.run(port=5000, debug=True)
