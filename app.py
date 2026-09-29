import av
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import WebRtcMode, webrtc_streamer

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
        font-size: 22px !important;
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
        font-size: 90px;
        text-align: center;
        background: rgba(0, 255, 204, 0.1);
        border: 2px solid #00ffcc;
        border-radius: 15px;
        padding: 10px;
        margin: 15px 0;
        box-shadow: 0 0 15px #00ffcc;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<marquee>★ MODO ULTRA-FLUIDO ACTIVADO ★ CERO LAG EN EL CYBER-ESPACIO ★</marquee>",
    unsafe_allow_html=True,
)

st.title("👾 EMOTION.EXE 👾")

st.markdown(
    """
<div class="y2k-box">
    <p>¡Cámara optimizada para respuesta inmediata sin retrasos!</p>
    <p style="font-size: 20px; color: #ff007f !important;">Emociones: Feliz 😊 | Triste 😢 | Enojado 😡 | Sorprendido 😲</p>
</div>
""",
    unsafe_allow_html=True,
)

emoji_display = st.empty()
text_display = st.empty()


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


# Procesador de video ultra-optimizado (Evita el lag procesando cada ciertos cuadros)
class OptimizedEmotionProcessor:

  def __init__(self):
    self.frame_count = 0
    self.last_emoji = "🤖"
    self.last_text = "ESPERANDO..."
    self.last_color = (255, 255, 255)

  def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    self.frame_count += 1

    # Analizamos la IA solo 1 de cada 10 fotogramas para liberar la carga de la red y eliminar el lag
    if self.frame_count % 10 == 0:
      try:
        resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        image_array = np.asarray(resized_img, dtype=np.float32)
        normalized_image_array = (image_array / 127.5) - 1.0
        data = np.expand_dims(normalized_image_array, axis=0)

        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]

        if " " in class_name:
          class_name = " ".join(class_name.split(" ")[1:])
        class_clean = class_name.lower().strip()

        if "feliz" in class_clean or "happy" in class_clean:
          self.last_emoji = "😁"
          self.last_color = (0, 255, 0)
        elif "triste" in class_clean or "sad" in class_clean:
          self.last_emoji = "😢"
          self.last_color = (255, 0, 0)
        elif "enojado" in class_clean or "angry" in class_clean:
          self.last_emoji = "😡"
          self.last_color = (0, 0, 255)
        elif "sorprendido" in class_clean or "surprised" in class_clean:
          self.last_emoji = "😲"
          self.last_color = (0, 255, 255)
        else:
          self.last_emoji = "🤖"
          self.last_color = (255, 255, 255)

        confidence = float(prediction[0][index]) * 100
        self.last_text = f"{class_name.upper()} ({confidence:.1f}%)"

        # Actualizar UI de Streamlit de forma segura
        emoji_display.markdown(
            f'<div class="emoji-container">{self.last_emoji}</div>',
            unsafe_allow_html=True,
        )
        text_display.markdown(
            f"<h3 style='color: #00ffcc !important;'>ESTADO:"
            f" {self.last_text}</h3>",
            unsafe_allow_html=True,
        )
      except Exception:
        pass

    # Dibujar texto fluido en el video sin ralentizar la transmisión
    cv2.putText(
        img,
        self.last_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        self.last_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        self.last_color,
        2,
        cv2.LINE_AA,
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Renderizar el componente de la cámara con configuración ligera
webrtc_streamer(
    key="y2k-emotion-cam-fluid",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=OptimizedEmotionProcessor,
    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #ff007f;'>⚡ STREAMING OPTIMIZADO -"
    " 0% LAG ⚡</p>",
    unsafe_allow_html=True,
)
