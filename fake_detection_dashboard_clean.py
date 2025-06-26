import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="🔍 Fake Detection Dashboard", layout="wide")

st.title("🔍 Fake Detection Dashboard")
st.sidebar.title("Navigation")
app_choice = st.sidebar.radio("Wähle ein Tool:", [
    "1️⃣ Zitat-KI-Prüfer",
    "2️⃣ Influencer-Fake-Check",
    "3️⃣ Bewertungs-Check",
    "4️⃣ Kleinanzeigen-Fakebild"
])

if app_choice == "1️⃣ Zitat-KI-Prüfer":
    st.header("📜 Fake-Zitat Checker")
    text = st.text_area("Füge ein Zitat oder Text ein:")
    if text:
        try:
            detector = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            result = detector(text)
            st.subheader("Erkennungs-Ergebnis:")
            st.json(result)
        except Exception as e:
            st.error(f"Fehler beim Laden des Modells oder Auswerten des Textes: {e}")

elif app_choice == "2️⃣ Influencer-Fake-Check":
    st.header("🕵️‍♀️ Influencer-Check: Echt oder KI-generiert?")
    uploaded = st.file_uploader("Profilbild hochladen", type=["jpg", "png"], key="influencer")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Profilbild")
        st.write("Reverse Image Search manuell:")
        st.markdown("👉 [Google Lens](https://lens.google.com/)")
        st.markdown("👉 [Yandex Bilder](https://yandex.com/images/)")
        st.markdown("👉 [TinEye](https://tineye.com/)")
        st.info("Tipp: Auch GAN-Detektor-Websites wie https://www.mayachitra.com/detect gan-face helfen.")

elif app_choice == "3️⃣ Bewertungs-Check":
    st.header("💬 Fake-Bewertung erkennen")
    text = st.text_area("Gib eine Bewertung ein:")
    if text:
        try:
            classifier = pipeline("text-classification", model="microsoft/xtremedistil-l6-h256-uncased")
            result = classifier(text)
            st.subheader("Sprachklassifikation:")
            st.json(result)
            st.info("Hinweis: Fake-Bewertungen sind oft übertrieben, vage oder werblich.")
        except Exception as e:
            st.error(f"Fehler beim Ausführen der Klassifikation: {e}")

elif app_choice == "4️⃣ Kleinanzeigen-Fakebild":
    st.header("📦 Kleinanzeigen-Fake-Bild-Check")
    uploaded = st.file_uploader("Anzeigenbild hochladen", type=["jpg", "jpeg", "png"], key="kleinanzeige")
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Hochgeladenes Bild")
        st.write("🔍 Suche dieses Bild auf:")
        st.markdown("👉 [Google Lens](https://lens.google.com/)")
        st.markdown("👉 [TinEye](https://tineye.com/)")