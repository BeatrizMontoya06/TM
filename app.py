import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2
from deepface import DeepFace
import av

# Configuración de la página
st.set_page_config(page_title="Detector de Emociones", page_icon="🎭")
st.title("Detección de Gestos en Tiempo Real 🎭")
st.write("Enciende tu cámara web para detectar si estás: **Feliz, Triste, Sorprendido o Enojado**.")

# Configuración para que WebRTC funcione correctamente en la nube (Streamlit Cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Cargar el modelo preentrenado de OpenCV para detectar rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Diccionario para traducir las emociones al español
TRADUCCION_EMOCIONES = {
    'happy': 'Feliz 😄',
    'sad': 'Triste 😢',
    'surprise': 'Sorprendido 😲',
    'angry': 'Enojado 😡',
    'neutral': 'Neutral 😐',
    'fear': 'Miedo 😨',
    'disgust': 'Disgusto 🤢'
}

def procesar_frame(frame):
    # Convertir el frame de WebRTC a formato de imagen de OpenCV (array de numpy)
    img = frame.to_ndarray(format="bgr24")
    
    # Convertir a escala de grises para mejorar la detección del rostro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Recortar solo la región del rostro
        face_img = img[y:y+h, x:x+w]
        
        try:
            # Analizar la emoción con DeepFace (enforce_detection=False evita errores si el rostro está borroso)
            resultados = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            
            # DeepFace puede devolver una lista si detecta múltiples caras, tomamos la primera
            emocion_dominante = resultados[0]['dominant_emotion']
            emocion_es = TRADUCCION_EMOCIONES.get(emocion_dominante, emocion_dominante)

            # Dibujar un rectángulo verde alrededor de la cara
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Escribir la emoción detectada arriba del rectángulo
            cv2.putText(img, emocion_es, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        except Exception as e:
            # Si falla el análisis de este frame, simplemente continuamos
            pass

    # Devolver la imagen procesada de vuelta al navegador
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Iniciar el componente de la cámara
webrtc_streamer(
    key="detector-emociones",
    video_frame_callback=procesar_frame,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False} # Solo necesitamos video
)

st.caption("Nota: La primera vez que se ejecute, el modelo tardará unos segundos en descargar los pesos de IA.")
