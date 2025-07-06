import streamlit as st
import feedparser
import requests
from PIL import Image
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
import json

# --------------------------
# Load CLIP AI model
# --------------------------
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_clip_model()

# --------------------------
# Define styles to detect
# --------------------------
STYLE_LABELS = ["boho", "minimalist", "scandinavian", "industrial"]

# --------------------------
# Load product recommendations
# --------------------------
@st.cache_data
def load_product_db():
    with open("product_db.json", "r") as f:
        return json.load(f)

product_db = load_product_db()

# --------------------------
# Use CLIP to classify style
# --------------------------
def classify_style(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    inputs = processor(text=STYLE_LABELS, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
    best_style = STYLE_LABELS[probs.argmax()]
    return best_style

# --------------------------
# Get image URLs from Pinterest RSS
# --------------------------
def fetch_pinterest_images(rss_url):
    feed = feedparser.parse(rss_url)
    image_urls = []
    for entry in feed.entries:
        if 'media_content' in entry:
            image_urls.append(entry.media_content[0]['url'])
        elif 'summary_detail' in entry and 'img' in entry.summary_detail['value']:
            start = entry.summary_detail['value'].find('img src=\"') + len('img src=\"')
            end = entry.summary_detail['value'].find('\"', start)
            image_urls.append(entry.summary_detail['value'][start:end])
    return image_urls[:10]

# --------------------------
# Recommend products by style
# --------------------------
def recommend_products(style):
    return [p for p in product_db if p['style'] == style]

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="MoodNest", page_icon="🪄")
st.title("🪄 MoodNest: AI-Powered Interior Design from Your Pinterest Style")

pinterest_input = st.text_input("Enter your public Pinterest Board URL:", "https://www.pinterest.com/username/boardname")

if st.button("Analyze My Board"):
    try:
        board_path = pinterest_input.split("pinterest.com/")[-1].strip("/")
        rss_url = f"https://www.pinterest.com/{board_path}.rss"
        st.write(f"Fetching images from: `{rss_url}`")

        image_urls = fetch_pinterest_images(rss_url)

        if not image_urls:
            st.warning("No images found. Make sure the board is public.")
        else:
            st.info("Detecting your interior design style from pins...")
            style_count = {}
            for url in image_urls:
                style = classify_style(url)
                style_count[style] = style_count.get(style, 0) + 1

            detected_style = max(style_count, key=style_count.get)
            st.success(f"🏡 Detected Style: **{detected_style.title()}**")

            st.markdown("### 🎁 Recommended Products")
            for product in recommend_products(detected_style):
                st.markdown(f"**{product['product_name']}**")
                st.image(product['image_url'], width=180)
                st.markdown(f"[{product['description']}]({product['affiliate_url']})")

            st.markdown("### 📌 Analyzed Pinterest Pins")
            for url in image_urls:
                st.image(url, width=120)

    except Exception as e:
        st.error(f"Error: {str(e)}")
