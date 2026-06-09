import streamlit as st
from PIL import Image
import time

# --- MOCK AI FUNCTION ---
# We will replace this with your actual PyTorch MobileNet inference later
def predict_disease(image):
    """Simulates a model prediction with a slight delay."""
    time.sleep(1.5) # Fake processing time
    
    # Mock result
    return {
        "class": "Tomato__Early_blight",
        "confidence": 94.2,
        "status": "Infected"
    }
# ------------------------

def main():
    # Page configuration
    st.set_page_config(page_title="FL Plant Disease Detection", page_icon="🌿", layout="centered")

    # Header
    st.title("🌿 Smart Agricultural Edge Node")
    st.write("Upload a leaf image to run a local inference check using the Federated Learning model.")
    st.markdown("---")

    # File Uploader
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 1. Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

        # 2. Add a button to trigger the analysis
        if st.button("Analyze Leaf", type="primary"):
            
            # Show a loading spinner while the "AI" thinks
            with st.spinner('Running local edge inference...'):
                result = predict_disease(image)
            
            # 3. Display the Results
            st.markdown("### 📊 Inference Results")
            
            # Using columns for a clean layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Status", value=result["status"])
            with col2:
                st.metric(label="Detected Class", value=result["class"].replace("_", " "))
            with col3:
                st.metric(label="Confidence", value=f"{result['confidence']}%")
                
            # Optional: Add a dynamic warning/success message
            if result["status"] == "Infected":
                st.error("Action Required: Disease detected. Please isolate the crop.")
            else:
                st.success("Crop appears healthy.")

if __name__ == "__main__":
    main()