import streamlit as st
import joblib
import base64
import random
import time
from datetime import datetime

# ================================
# 🎯 CONFIGURATION
# ================================
MODEL_PATH = "restaurant_feedback_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# ================================
# ⚙️ UTILITIES

# ================================
def get_base64_image_url(uploaded_file):
    """Converts uploaded file to Base64 URL."""
    try:
        bytes_data = uploaded_file.getvalue()
        base64_encoded_data = base64.b64encode(bytes_data).decode("utf-8")
        mime_type = uploaded_file.type or "image/png"
        return f"data:{mime_type};base64,{base64_encoded_data}"
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def set_cinematic_bg(base64_urls, interval_per_image=6):
    """Applies cinematic slideshow background."""
    num_images = len(base64_urls)
    total_duration = num_images * interval_per_image
    OVERLAY_OPACITY = "rgba(0,0,0,0.6)"

    FROSTED_GLASS_SELECTORS = """
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stTabs"] > div:nth-child(2)
    """

    if num_images == 0:
        st.info("No images uploaded. Using default gradient background.")
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #1b2735, #090a0f);
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        return

    css_keyframes = []
    for i in range(num_images):
        start_percent = (i * 100) / num_images
        hold_percent = start_percent + (100 / num_images)
        css_keyframes.append(f"{start_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
        css_keyframes.append(f"{hold_percent:.2f}% {{ background-image: url('{base64_urls[i]}'); }}")
    css_keyframes.append(f"100% {{ background-image: url('{base64_urls[0]}'); }}")

    st.markdown(f"""
        <style>
        .stApp {{
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-image: url('{base64_urls[0]}');
            animation: cinematicBg {total_duration}s infinite;
            color: white;
        }}
        @keyframes cinematicBg {{
            {"".join(css_keyframes)}
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: {OVERLAY_OPACITY};
            z-index: 0;
        }}
        {FROSTED_GLASS_SELECTORS} {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(8px);
            border-radius: 16px;
            padding: 20px;
            z-index: 10;
        }}
        * {{ font-family: 'Poppins', sans-serif; }}
        [data-testid="stHeader"], [data-testid="stToolbar"] {{ background: transparent !important; }}
        </style>
    """, unsafe_allow_html=True)

# ================================
# 🧠 LOAD MODEL
# ================================
try:
    svm = joblib.load(MODEL_PATH)
    tfidf = joblib.load(VECTORIZER_PATH)
    st.info("✅ Model and Vectorizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model/vectorizer: {e}")
    svm, tfidf = None, None

# ================================
# 📂 SIDEBAR
# ================================
base64_image_urls = []
with st.sidebar:
    st.title("⚙️ App Configuration")
    uploaded_files = st.file_uploader(
        "🖼️ Upload background images (slideshow):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload 3+ background images for cinematic transitions."
    )
    if uploaded_files:
        with st.spinner("Processing images..."):
            for file in uploaded_files:
                url = get_base64_image_url(file)
                if url:
                    base64_image_urls.append(url)
            time.sleep(0.5)
        st.success("✅ Images processed successfully!")

    st.markdown("---")
    st.subheader("📚 Model Info")
    st.info("This app analyzes restaurant customer feedback and predicts sentiment using a trained **SVM** model with TF-IDF vectorization.")
    st.markdown(f"📅 Last Updated: **{datetime.now().strftime('%b %d, %Y')}**")
    st.markdown("Made with ❤️ ", unsafe_allow_html=True)
    st.markdown("✨ Developed by **Umar Imam**", unsafe_allow_html=True)

set_cinematic_bg(base64_image_urls)

# ================================
# 🍽️ HEADER
# ================================
st.set_page_config(layout="centered")
st.markdown("""
<h1 style='text-align:center; color:#ff7b00; text-shadow: 2px 2px 6px #000;'>🍽️ Restaurant Customer Feedback Analyzer</h1>
<p style='text-align:center; font-size:18px; color:#fff;'>Understand how your customers feel through AI-driven sentiment analysis.</p>
""", unsafe_allow_html=True)

# ================================
# 📊 TABS
# ================================
tab1, tab2 = st.tabs(["💬 Analyze Feedback", "ℹ️ Model Info"])

# ================================
# 🔮 TAB 1 — FEEDBACK ANALYSIS
# ================================
with tab1:
    st.header("Enter Customer Feedback")
    feedback_text = st.text_area("📝 Type or paste feedback below:", placeholder="e.g., The food was amazing and the staff were friendly!")

    if st.button("🚀 Analyze Sentiment", use_container_width=True):
        if not svm or not tfidf:
            st.error("⚠️ Model not loaded. Please check your .pkl files.")
        elif not feedback_text.strip():
            st.warning("⚠️ Please enter some feedback text.")
        else:
            with st.spinner("Analyzing sentiment..."):
                # Vectorize user input
                vectorized_text = tfidf.transform([feedback_text])

                # Make prediction
                prediction_num = svm.predict(vectorized_text)[0]

                # Map numeric predictions to sentiment labels
                label_map = {
                    0: "Negative",
                    1: "Positive",
                    2: "Neutral"
                }

                prediction = label_map.get(prediction_num, "Neutral").lower()
                prediction_str = str(prediction).lower()

                st.balloons()

                color_map = {
                    "positive": "#00C851",
                    "negative": "#ff4444",
                    "neutral": "#33b5e5"
                }

                witty_lines = {
                    "positive": [
                        "Keep delighting your customers! 🌟",
                        "Your customers are loving it — keep the momentum! 💖",
                        "Fantastic feedback — your efforts are paying off! 🎉"
                    ],
                    "negative": [
                        "Time to spice things up a bit! 🌶️",
                        "Some customers are unhappy — let’s fix that 💪",
                        "Don't worry, feedback is the first step to improvement! 🚀"
                    ],
                    "neutral": [
                        "Mixed feelings — maybe consistency is key 🔄",
                        "Neither hot nor cold — you can warm things up! ☕",
                        "Good, but there’s room to impress even more ✨"
                    ]
                }

                st.markdown(f"""
                    <div style="background-color:{color_map.get(prediction_str, '#444')};
                                padding:25px; border-radius:15px;
                                text-align:center;
                                box-shadow:0 0 20px {color_map.get(prediction_str, '#444')};">
                        <h2 style="color:white;">Predicted Sentiment</h2>
                        <h1 style="color:white; font-size:3.5em;">{prediction_str.capitalize()}</h1>
                    </div>
                """, unsafe_allow_html=True)

                message = random.choice(
                    witty_lines.get(prediction_str, ["Keep striving to make every customer smile! 😄"])
                )
                st.info(message)



# ================================
# 📘 TAB 2 — MODEL INFO
# ================================
with tab2:
    st.header("Model Overview & Performance")
    st.info("**Algorithm:** Support Vector Machine (SVM) — trained on restaurant feedback dataset.")
    st.markdown("""
    **Training Data:**  
    - Input Feature: `Customer Feedback` (text)  
    - Output Label: `Sentiment` (Positive / Negative / Neutral)
    
    The model uses TF-IDF vectorization and SVM classification to determine the overall emotion behind customer reviews.
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    col1.metric("Model Type", "SVM Classifier")
    col2.metric("Vectorization", "TF-IDF")

if __name__ == "__main__":
    pass
