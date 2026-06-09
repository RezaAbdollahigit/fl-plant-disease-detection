import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.mobilenet import get_mobilenet
import os

# 1. Dataset Classes
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# 2. Load ALL Models (Cached so we don't crash your RAM)
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES)
    
    def load_specific_model(filename):
        model = get_mobilenet(num_classes=num_classes, pretrained=False)
        path = os.path.join('results', filename)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        else:
            st.error(f"Missing model file: {path}. Did you run the simulation?")
        model.to(device)
        model.eval()
        return model

    # Load the 3 distinct brains
    baseline_model = load_specific_model('baseline_model.pth')
    fedavg_model = load_specific_model('fedavg_model.pth')
    fedprox_model = load_specific_model('fedprox_model.pth')
    
    return baseline_model, fedavg_model, fedprox_model, device

baseline, fedavg, fedprox, device = load_all_models()

# 3. Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# 4. Standardized Inference Function
def get_prediction(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    raw_class = CLASS_NAMES[predicted_idx.item()]
    clean_name = raw_class.replace("___", " - ").replace("__", " ").replace("_", " ")
    status = "Healthy" if "healthy" in raw_class.lower() else "Infected"
    
    return clean_name, round(confidence.item() * 100, 2), status

# ==========================================
# 🖥️ UI LAYOUT & DASHBOARD
# ==========================================
def main():
    st.set_page_config(page_title="FL Defense Dashboard", page_icon="🌿", layout="wide")

    st.title("🌿 Federated Learning vs. Centralized AI")
    st.write("Upload a single edge-device image to compare the performance of standard FedAvg against the FedProx proximal penalty under heavily isolated (Non-IID) data conditions.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose a leaf image from the edge device...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col_img, col_empty = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

        if st.button("Run Comparative Inference", type="primary"):
            tensor = preprocess_image(image)
            
            with st.spinner('Running multi-model GPU inference...'):
                base_class, base_conf, base_stat = get_prediction(baseline, tensor)
                avg_class, avg_conf, avg_stat = get_prediction(fedavg, tensor)
                prox_class, prox_conf, prox_stat = get_prediction(fedprox, tensor)
            
            st.markdown("### 📊 Live Model Comparison")
            
            # The 3-Way Split Screen
            col1, col2, col3 = st.columns(3)
            
            # Column 1: The Gold Standard
            with col1:
                st.info("**1. Centralized Baseline**\n\n*(The 'Gold Standard' with access to all data)*")
                st.metric(label="Predicted Disease", value=base_class)
                st.metric(label="Confidence", value=f"{base_conf}%")
                
            # Column 2: The Broken System
            with col2:
                st.warning("**2. Standard FedAvg**\n\n*(Suffers from Catastrophic Forgetting)*")
                st.metric(label="Predicted Disease", value=avg_class)
                st.metric(label="Confidence", value=f"{avg_conf}%", delta="- Broken / Confused" if avg_class != base_class else "Agrees with Baseline", delta_color="inverse" if avg_class != base_class else "normal")
                
            # Column 3: The Solution
            with col3:
                st.success("**3. FedProx (Proposed)**\n\n*(Stabilized via Proximal Penalty)*")
                st.metric(label="Predicted Disease", value=prox_class)
                st.metric(label="Confidence", value=f"{prox_conf}%", delta="Corrected Alignment" if prox_class == base_class and avg_class != base_class else None)

if __name__ == "__main__":
    main()