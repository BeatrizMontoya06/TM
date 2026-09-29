import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from PIL import Image

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Y2K Emotion Detector 👾", page_icon="💖", layout="centered"
)

# --- ESTÉTICA Y2K (CSS CUSTOMIZADO) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0f051d 0%, #2b0b3f 50%, #121c3a 100%);
        color: #00ffcc;
        font-family: 'VT323', monospace;
    }

    h1, h2, h3 {
        font-family: 'Press Start 2P', cursive !important;
        color: #ff007f !important;
        text-shadow: 3px 3px #00ffcc;
        font-size: 20px !important;
        text-align: center;
    }

    p, li, label {
        font-family: 'VT323', monospace !important;
        font-size: 26px !important;
        color: #e0ffff !important;
    }

    .y2k-box {
        border: 3px dashed #ff007f;
        background: rgba(18, 5, 35, 0.85);
        padding: 20px;
        box-shadow: 0px 0px 20px #ff007f;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }

    marquee {
        font-family: 'Press Start 2P', cursive;
        color: #ffff00;
        background: #ff007f;
        padding: 5px;
        font-size: 12px;
        border: 2px solid #00ffcc;
        margin-bottom: 15px;
    }

    .emoji-container {
        font-size: 100px;
        text-align: center;
        background: rgba(0, 255, 204, 0.1);
        border: 2px solid #00ffcc;
        border-radius: 15px;
        padding: 15px;
        margin: 15px 0;
        box-shadow: 0 0 20px #00ffcc;
    }
    
    .stButton>button {
        background-color: #ff007f !important;
        color: #ffffff !important;
        font-family: 'Press Start 2P', cursive !important;
        font-size: 14px !important;
        border: 3px solid #00ffcc !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        width: 100%;
        box-shadow: 0 0 10px #ff007f;
    }
    .stButton>button:hover {
        background-color: #00ffcc !important;
        color: #0f051d !important;
        border: 3px solid #ff007f !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<marquee>★ DETECTOR DE ROSTRO INTELIGENTE ACTIVADO ★ ENFOCA TU CARA ★</marquee>",
    unsafe_allow_html=True,
)

st.title("👾 EMOTION.EXE 👾")

st.markdown(
    """
<div class="y2k-box">
    <p>¡El sistema detectará tu rostro automáticamente para evaluar tu expresión!</p>
    <p style="font-size: 20px; color: #ff007f !important;">Emociones: Feliz 😊 | Triste 😢 | Enojado 😡 | Sorprendido 😲</p>
</div>
""",
    unsafe_allow_html=True,
)


# Cargar el modelo con caché
@st.cache_resource
def load_emotion_model():
  model = load_model("keras_model.h5", compile=False)
  with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]
  return model, class_names


try:
  model, class_names = load_emotion_model()
except Exception as e:
  st.error(f"⚠️ Error al cargar el modelo: {e}")

# Widget para capturar foto
picture = st.camera_input("📸 TOMA UNA FOTO DE TU ROSTRO")

if picture is not None:
  # Cargar imagen usando PIL y convertirla a formato OpenCV (BGR)
  image_pil = Image.open(picture)
  img_cv = np.array(image_pil)
  img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

  # Cargar el clasificador de rostros integrado en OpenCV
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
  )
  gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

  # Buscar rostros en la foto
  faces = face_cascade.detectMultiScale(
      gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
  )

  # Si encuentra al menos una cara, recorta exactamente esa región
  if len(faces) > 0:
    # Tomar el primer rostro detectado (x, y, w, h)
    x, y, w, h = faces[0]
    # Expandir un poco el margen del recorte para incluir bien la expresión
    margin = int(w * 0.1)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img_cv.shape[1], x + w + margin)
    y2 = min(img_cv.shape[0], y + h + margin)

    face_crop = img_cv[y1:y2, x1:x2]
    # Convertir de vuelta a RGB para el modelo
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    image_to_predict = Image.fromarray(face_rgb)
  else:
    # Si por alguna razón el filtro no detecta la cara de forma exacta, usa la imagen completa
    image_to_predict = image_pil
    st.warning(
        "⚠️ No se detectó un rostro claramente, analizando la imagen"
        " completa..."
    )

  # Preprocesar para Teachable Machine (224x224 exactos)
  image_resized = image_to_predict.resize((224, 224))
  image_array = np.asarray(image_resized, dtype=np.float32)
  normalized_image_array = (image_array / 127.5) - 1.0
  data = np.expand_dims(normalized_image_array, axis=0)

  # Predicción con la IA
  prediction = model.predict(data, verbose=0)
  index = np.argmax(prediction)
  class_name = class_names[index]

  # Limpiar etiqueta
  if " " in class_name:
    class_name = " ".join(class_name.split(" ")[1:])
  class_clean = class_name.lower().strip()

  # Mapear emojis
  if "feliz" in class_clean or "happy" in class_clean:
    emoji = "😁"
  elif "triste" in class_clean or "sad" in class_clean:
    emoji = "😢"
  elif "enojado" in class_clean or "angry" in class_clean:
    emoji = "😡"
  elif "sorprendido" in class_clean or "surprised" in class_clean:
    emoji = "😲"
  else:
    emoji = "🤖"

  confidence = float(prediction[0][index]) * 100

  # Mostrar resultados limpios
  st.markdown("---")
  st.markdown("<h3>⚡ RESULTADO DEL ROSTRO ⚡</h3>", unsafe_allow_html=True)
  st.markdown(
      f'<div class="emoji-container">{emoji}</div>', unsafe_allow_html=True
  )
  st.markdown(
      f"<h2 style='text-align: center; color: #00ffcc !important;'>{class_name.upper()}"
      f" - {confidence:.1f}%</h2>",
      unsafe_allow_html=True,
  )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #ff007f;'>⚡ SYSTEM ONLINE -"
    " POWERED BY STREAMLIT & GITHUB ⚡</p>",
    unsafe_allow_html=True,
)
