import cv2
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

# Configuration de l'interface
st.set_page_config(page_title="IA Détecteur", layout="wide")
st.title("🚀 Système de Détection d'Individus")

# Chargement du modèle IA
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Barre latérale pour les réglages
seuil = st.sidebar.slider("Seuil de détection", 0.0, 1.0, 0.5)
run = st.sidebar.checkbox('Lancer la caméra', value=True)

# Zone d'affichage vidéo
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0) # 0 pour la webcam intégrée

while run:
    success, frame = cap.read()
    if not success:
        st.error("Caméra non détectée")
        break
    
    # Conversion pour l'IA
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection
    results = model(img, conf=seuil, classes=0, verbose=False)
    
    # Dessiner les résultats
    annotated_frame = results[0].plot()
    
    # Affichage dans l'app
    FRAME_WINDOW.image(annotated_frame)
    
    # Statistiques en direct
    count = len(results[0].boxes)
    st.sidebar.metric("Personnes détectées", count)

cap.release()
