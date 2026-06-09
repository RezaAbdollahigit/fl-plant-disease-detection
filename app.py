import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.mobilenet import get_mobilenet

# 1. Define the exact class names from your dataset
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# 2. Load the Trained Model (Cached for performance)
@st.cache_resource
def load_model():
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the architecture
    model = get_mobilenet(num_classes=len(CLASS_NAMES), pretrained=False)
    
    # Load your trained weights
    model.load_state_dict(torch.load('results/baseline_model.pth', map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set to evaluation mode
    return model, device

model, device = load_model()

# 3. Image Preprocessing
def preprocess_image(image):
    """Transforms the uploaded PIL Image into a PyTorch tensor."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB') # Ensure it has 3 color channels
    # Add a batch dimension at the front: shape becomes [1, 3, 224, 224]
    return transform(image).unsqueeze(0).to(device)

# 4. Real AI Inference
def predict_disease(image):
    """Passes the image through MobileNetV2 and calculates confidence."""
    tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(tensor)
        # Apply Softmax to convert raw output numbers into percentages (0 to 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    raw_class_name = CLASS_NAMES[predicted_idx.item()]
    
    # Clean up the folder name for a professional UI display
    clean_name = raw_class_name.replace("___", " - ").replace("__", " ").replace("_", " ")
    status = "Healthy" if "healthy" in raw_class_name.lower() else "Infected"
    
    return {
        "class": clean_name,
        "confidence": round(confidence.item() * 100, 2),
        "status": status
    }

# ---------------------------------------------------------
# UI LAYOUT
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="FL Plant Disease Detection", page_icon="🌿", layout="centered")

    st.title("🌿 Smart Agricultural Edge Node")
    st.write("Upload a leaf image to run a local inference check using the PyTorch model.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

        if st.button("Analyze Leaf", type="primary"):
            
            with st.spinner('Running AI Inference on GPU...'):
                result = predict_disease(image)
            
            st.markdown("### 📊 Inference Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Status", value=result["status"])
            with col2:
                st.metric(label="Detected Class", value=result["class"])
            with col3:
                st.metric(label="Confidence", value=f"{result['confidence']}%")
                
            if result["status"] == "Infected":
                st.error("⚠️ Action Required: Disease detected. Please isolate the crop.")
            else:
                st.success("✅ Crop appears healthy.")

if __name__ == "__main__":
    main()