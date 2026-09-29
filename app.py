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
    
    /* Estilizar el botón de Streamlit para que parezca de videojuego retro */
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

# Marquesina animada estilo 2000s
st.markdown(
    "<marquee>★ MODO FOTO INSTANTÁNEA ACTIVADO ★ CERO LAG ★ CAPTURA TU EMOCIÓN ★</marquee>",
    unsafe_allow_html=True,
)

st.title("👾 EMOTION.EXE 👾")

st.markdown(
    """
<div class="y2k-box">
    <p>¡Toma una foto haciendo una emoción para que el sistema la detecte al instante!</p>
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
  st.error(
      f"⚠️ ERROR CRÍTICO EN EL SISTEMA: No se pudo cargar el modelo. Detalle:"
      f" {e}"
  )

# Componente nativo de Streamlit para tomar foto con la cámara del dispositivo
picture = st.camera_input("📸 ENCIENDE TU CÁMARA Y TOMA UNA FOTO")

if picture is not None:
  # Cargar la imagen tomada por el usuario
  image = Image.open(picture)

  # Preprocesar la imagen para Teachable Machine
  # Convertir a RGB por si acaso y redimensionar exactamente a 224x224
  image_resized = image.resize((224, 224))
  image_array = np.asarray(image_resized, dtype=np.float32)

  # Normalizar igual que en el entrenamiento (-1 a 1)
  normalized_image_array = (image_array / 127.5) - 1.0
  data = np.expand_dims(normalized_image_array, axis=0)

  # Realizar predicción con el modelo Keras
  prediction = model.predict(data, verbose=0)
  index = np.argmax(prediction)
  class_name = class_names[index]

  # Limpiar el nombre de la clase (por si tiene índices al inicio tipo "0 feliz")
  if " " in class_name:
    class_name = " ".join(class_name.split(" ")[1:])
  class_clean = class_name.lower().strip()

  # Mapear a Emojis de forma precisa
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

  # Mostrar resultados con estilo Y2K impactante
  st.markdown("---")
  st.markdown("<h3>⚡ RESULTADO DEL ANÁLISIS CIBERNÉTICO ⚡</h3>", unsafe_allow_html=True)
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
