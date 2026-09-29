from PIL import Image as Image, ImageOps as ImagOps
import cv2
from keras.models import load_model
import numpy as np
import platform
import streamlit as st

# ─────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA (ESTILO ARCADE RETRO)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Detección de objetos by bee",
    page_icon="🕹️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# ESTILOS CSS: ESTÉTICA ARCADE / RETRO 80s / PIXEL
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&display=swap');

    /* Fondo general estilo máquina recreativa oscura con scanlines */
    .stApp {
        background-color: #0b061a !important;
        background-image: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06)) !important;
        background-size: 100% 4px, 6px 100% !important;
        color: #00ffcc !important;
        font-family: 'VT323', monospace !important;
    }

    /* Ocultar elementos nativos molestos de streamlit */
    #MainMenu, footer, header {visibility: hidden;}

    /* Marquesina Estilo Arcade */
    .arcade-marquee {
        background: linear-gradient(180deg, #ff007f 0%, #7b00ff 100%);
        border: 4px solid #ffff00;
        box-shadow: 0px 0px 20px #ff007f, inset 0px 0px 10px #ffff00;
        text-align: center;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 4px;
    }

    .arcade-title {
        font-family: 'Press Start 2P', monospace !important;
        font-size: 1.6rem !important;
        color: #ffff00 !important;
        text-shadow: 3px 3px #ff007f, 6px 6px #000000;
        margin: 0;
        line-height: 1.5;
    }

    .arcade-subtitle {
        font-family: 'VT323', monospace !important;
        font-size: 1.5rem !important;
        color: #00ffcc !important;
        margin-top: 8px;
        text-shadow: 2px 2px #000000;
        letter-spacing: 2px;
    }

    /* Contenedores tipo Gabinete / Tarjeta */
    .arcade-cabinet {
        background: #150b2e;
        border: 3px dashed #00ffcc;
        box-shadow: 5px 5px 0px #ff007f;
        padding: 20px;
        margin-bottom: 20px;
    }

    /* Estilo para los Headers de resultados estilo Score / Game Over */
    .arcade-result-left {
        background: #2b0018;
        border: 3px solid #ff007f;
        color: #ff007f;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.9rem !important;
        padding: 15px;
        text-align: center;
        box-shadow: 4px 4px 0px #ffff00;
        margin-top: 15px;
    }

    .arcade-result-up {
        background: #002b1f;
        border: 3px solid #00ffcc;
        color: #00ffcc;
        font-family: 'Press Start 2P', monospace;
        font-size: 0.9rem !important;
        padding: 15px;
        text-align: center;
        box-shadow: 4px 4px 0px #ff007f;
        margin-top: 15px;
    }

    /* Botones de Cámara Arcade */
    button[data-testid="stBaseButton-secondary"] {
        background-color: #ffff00 !important;
        color: #000000 !important;
        border: 3px solid #ff007f !important;
        font-family: 'Press Start 2P', monospace !important;
        font-size: 0.8rem !important;
        box-shadow: 4px 4px 0px #00ffcc !important;
        border-radius: 0px !important;
    }
    button[data-testid="stBaseButton-secondary"]:hover {
        background-color: #ff007f !important;
        color: #ffff00 !important;
        box-shadow: 4px 4px 0px #ffff00 !important;
    }

    /* Sidebar Temático */
    [data-testid="stSidebar"] {
        background-color: #0d061c !important;
        border-right: 4px solid #ff007f !important;
    }
    [data-testid="stSidebar"] * {
        color: #ffff00 !important;
        font-family: 'VT323', monospace !important;
        font-size: 1.4rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# CARGA DEL MODELO Y CONFIGURACIÓN INICIAL
# ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
  return load_model("keras_model.h5")


model = load_keras_model()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# ─────────────────────────────────────────────
# MARQUESINA SUPERIOR (HEADER)
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="arcade-marquee">
    <h1 class="arcade-title">Detección de objetos<br><span style="color: #00ffcc;">by bee</span></h1>
    <div class="arcade-subtitle">★ INSERT COIN TO START NEURAL VISION ★</div>
</div>
""",
    unsafe_allow_html=True,
)

# Información de sistema en versión retro
col_info1, col_info2 = st.columns([2, 1])
with col_info1:
  st.markdown(
      f"<span style='color: #ffff00; font-family: VT323; font-size: 1.3rem;'>[SYS_VER]: Python {platform.python_version()}</span>",
      unsafe_allow_html=True,
  )
with col_info2:
  st.markdown(
      "<span style='color: #ff007f; font-family: VT323; font-size: 1.3rem; float:right;'>CREDITS: [ 01 ]</span>",
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SECCIÓN DE IMAGEN DE REFERENCIA / MUESTRA
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="arcade-cabinet">
    <div style="font-family: 'Press Start 2P'; font-size: 0.8rem; color: #ffff00; margin-bottom: 10px;">STAGE 0: REFERENCE ASSET</div>
""",
    unsafe_allow_html=True,
)

try:
  image = Image.open("OIG5.jpg")
  st.image(image, width=350)
except Exception:
  st.warning(
      "⚠️ [WARNING]: Imagen de referencia 'OIG5.jpg' no encontrada en el"
      " directorio."
  )

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BARRA LATERAL (SIDEBAR) ESTILO INVENTARIO ARCADE
# ─────────────────────────────────────────────
with st.sidebar:
  st.markdown("### 🕹️ POWER-UP INSTRUCTIONS")
  st.markdown(
      "Usando un modelo entrenado en **Teachable Machine**, puedes usar esta app"
      " de máquina arcade para identificar posturas o gestos en tiempo real."
  )
  st.markdown("---")
  st.markdown("<b>CONTROLS:</b><br>• Conecta tu cámara.<br>• Posiciónate.", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CAPTURA DE CÁMARA (WIDGET PRINCIPAL)
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="arcade-cabinet" style="border-color: #ff007f; box-shadow: 5px 5px 0px #00ffcc;">
    <div style="font-family: 'Press Start 2P'; font-size: 0.8rem; color: #ff007f; margin-bottom: 10px;">STAGE 1: CAMERA SCANNER</div>
""",
    unsafe_allow_html=True,
)

img_file_buffer = st.camera_input("PLAYER 1: TOMA UNA FOTO")

st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LÓGICA DE INFERENCIA Y PREDICCIÓN (MANTENIDA ÍNTEGRA)
# ─────────────────────────────────────────────
if img_file_buffer is not None:
  # To read image file buffer with OpenCV:
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
  # To read image file buffer as a PIL Image:
  img = Image.open(img_file_buffer)

  newsize = (224, 224)
  img = img.resize(newsize)
  # To convert PIL Image to numpy array:
  img_array = np.array(img)

  # Normalize the image
  normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
  # Load the image into the array
  data[0] = normalized_image_array

  # run the inference
  prediction = model.predict(data)
  print(prediction)

  # Salidas de resultados estilizadas con contenedores arcade adaptados a tu lógica
  if prediction[0][0] > 0.5:
    st.markdown(
        f"""
        <div class="arcade-result-left">
            ◀ DETECCIÓN: IZQUIERDA<br>
            <span style="font-size: 0.7rem; color: #ffff00;">PROBABILIDAD: {str(prediction[0][0])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

  if prediction[0][1] > 0.5:
    st.markdown(
        f"""
        <div class="arcade-result-up">
            ▲ DETECCIÓN: ARRIBA<br>
            <span style="font-size: 0.7rem; color: #ffff00;">PROBABILIDAD: {str(prediction[0][1])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

  # if prediction[0][2]>0.5:
  #     st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))

# ─────────────────────────────────────────────
# PIE DE PÁGINA RETRO
# ─────────────────────────────────────────────
st.markdown("<br><hr style='border: 2px dashed #ff007f;'>", unsafe_allow_html=True)
st.markdown(
    """
<div style="text-align: center; font-family: 'VT323'; color: #ffff00; font-size: 1.4rem;">
    © 198X-2026 // Detección de objetos by bee — ALL RIGHTS RESERVED
</div>
""",
    unsafe_allow_html=True,
)
