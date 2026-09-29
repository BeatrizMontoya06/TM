import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# --- CONFIGURACIÓN DE LA PÁGINA Y ESTILO Y2K (AÑOS 2000) ---
st.set_page_config(page_title="CyberPortal 2000 - Gestures v1.0", page_icon="👾", layout="wide")

# Inyección de CSS personalizado con estética 2000s (Neón, verde matriz/cyber, fuentes estilo terminal retro)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=VT323&family=Orbitron:wght@400;700&display=swap');

    .stApp {
        background-color: #0b0c10;
        color: #66fcf1;
        font-family: 'Orbitron', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'VT323', monospace !important;
        color: #45a29e !important;
        text-shadow: 2px 2px #1f2833;
        letter-spacing: 2px;
    }

    .sidebar .stSidebar {
        background-color: #1f2833 !important;
        border-right: 2px dashed #45a29e;
    }

    div.stButton > button {
        background: linear-gradient(45deg, #1f2833, #45a29e);
        color: #66fcf1;
        border: 2px solid #66fcf1;
        font-family: 'VT323', monospace;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0px 0px 10px #45a29e;
        border-radius: 0px;
    }
    
    div.stButton > button:hover {
        background: #66fcf1;
        color: #0b0c10;
        box-shadow: 0px 0px 20px #66fcf1;
    }

    .retro-box {
        border: 2px solid #45a29e;
        padding: 15px;
        background-color: #1f2833;
        box-shadow: 4px 4px 0px #66fcf1;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CARGAR MODELO Y CONFIGURAR DATOS ---
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

try:
    model = cargar_modelo()
except Exception as e:
    st.error("⚠️ ADVERTENCIA: No se encontró el archivo 'keras_model.h5'. Asegúrate de subirlo a la misma carpeta.")

# --- DISEÑO Y REORGANIZACIÓN (ESTILO 2000) ---
st.title("👾 CYBER_GESTURE_SCANNER_2000.EXE 👾")
st.markdown("---")

# Panel lateral con estética de la vieja escuela
with st.sidebar:
    st.subheader("🌐 SISTEMA OPERATIVO v2.0")
    st.write("Bienvenido al portal de reconocimiento de movimiento basado en Teachable Machine.")
    st.markdown("---")
    st.info(f"SYS_INFO: Python v{platform.python_version()}")
    st.markdown("Conecta tu webcam, haz una pose y deja que la red neuronal decida tu destino digital.")
    
    # Intentar mostrar la imagen decorativa si existe, si no, omitir
    try:
        side_image = Image.open('OIG5.jpg')
        st.image(side_image, caption="[ SYSTEM_AVATAR ]", use_container_width=True)
    except:
        st.warning("[!] Imagen OIG5.jpg no detectada.")

# Reorganización principal: Dividir en columnas para una interfaz más moderna pero con alma retro
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### [1] CAPTURA DE VÍDEO EN VIVO")
    st.markdown('<div class="retro-box">', unsafe_allow_html=True)
    img_file_buffer = st.camera_input("Activar Sensor Óptico (Cámara)")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("### [2] ANÁLISIS DE INTELIGENCIA ARTIFICIAL")
    st.markdown('<div class="retro-box">', unsafe_allow_html=True)
    
    if img_file_buffer is not None:
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        img = Image.open(img_file_buffer)
        
        # Redimensionar y procesar la imagen
        newsize = (224, 224)
        img = img.resize(newsize)
        img_array = np.array(img)
        
        # Normalizar la imagen para Keras
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        
        # Ejecutar inferencia
        prediction = model.predict(data)
        
        st.write("---")
        st.markdown("**ESTADO DEL ESCANEO:**")
        
        # Evaluar predicciones del modelo con umbrales visuales estilo retro
        detectado = False
        if prediction[0][0] > 0.5:
            st.success(f'🟢 DIRECCIÓN DETECTADA: **IZQUIERDA**')
            st.progress(float(prediction[0][0]))
            st.write(f'Probabilidad de acierto: `{prediction[0][0]*100:.2f}%`')
            detectado = True
            
        if prediction[0][1] > 0.5:
            st.success(f'🔵 DIRECCIÓN DETECTADA: **ARRIBA**')
            st.progress(float(prediction[0][1]))
            st.write(f'Probabilidad de acierto: `{prediction[0][1]*100:.2f}%`')
            detectado = True
            
        if not detectado:
            st.warning("⚠️ Patrón no reconocido claramente. Intenta ajustar tu posición.")
            
    else:
        st.info("Esperando entrada de imagen... Presiona el botón de la cámara para iniciar el escaneo.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Pie de página retro
st.markdown("---")
st.markdown("<p style='text-align: center; color: #45a29e; font-family: VT323, monospace; font-size: 18px;'>[CYBER_NET 2000 — TODOS LOS DERECHOS RESERVADOS]</p>", unsafe_allow_html=True)
