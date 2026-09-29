import av
import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Detector de Emociones", page_icon="😊", layout="centered"
)

st.title("🧠 Detector de Emociones en Tiempo Real")
st.write(
    "Esta aplicación utiliza tu cámara web y un modelo de Inteligencia Artificial para detectar 4 emociones: **feliz, triste, enojado y sorprendido**."
)


# Cargar el modelo y las etiquetas con caché para optimizar el rendimiento
@st.cache_resource
def load_emotion_model():
  # Deshabilitar la compilación para evitar problemas de compatibilidad con versiones de Keras/TensorFlow
  model = load_model("keras_model.h5", compile=False)
  with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines()]
  return model, class_names


try:
  model, class_names = load_emotion_model()
except Exception as e:
  st.error(
    f"Error al cargar el modelo o las etiquetas. Asegúrate de subir 'keras_model.h5' y 'labels.txt'. Detalle: {e}"
  )


# Procesador de video para WebRTC
class EmotionProcessor:

  def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    # Redimensionar la imagen a 224x224 (tamaño estándar requerido por Teachable Machine)
    resized_img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    image_array = np.asarray(resized_img, dtype=np.float32)

    # Normalizar la imagen exactamente como lo hace Teachable Machine (-1 a 1)
    normalized_image_array = (image_array / 127.5) - 1.0

    # Crear la matriz de entrada para la predicción
    data = np.expand_dims(normalized_image_array, axis=0)

    # Hacer la predicción
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index]
    # Limpiar el nombre de la clase (quita números de índice si los tiene, ej: "0 Feliz" -> "Feliz")
    if " " in class_name:
      class_name = " ".join(class_name.split(" ")[1:])
    confidence_score = float(prediction[0][index])

    # Dibujar el resultado en el fotograma de video que se muestra en pantalla
    color = (0, 255, 0)
    if "enojado" in class_name.lower():
      color = (0, 0, 255)
    elif "triste" in class_name.lower():
      color = (255, 0, 0)
    elif "sorprendido" in class_name.lower():
      color = (0, 255, 255)

    text = f"{class_name} ({confidence_score * 100:.1f}%)"
    cv2.putText(
        img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
    )

    return av.VideoFrame.from_ndarray(img, format="bgr24")


# Configurar el componente de transmisión de la cámara web
webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

st.markdown("---")
st.markdown("### 📋 Instrucciones:")
st.markdown("1. Haz clic en **START** para encender la cámara.")
st.markdown(
    "2. Concede los permisos de tu navegador para acceder a la cámara web."
)
st.markdown(
    "3. Colócate frente a la cámara y expresa una de las 4 emociones."
)
