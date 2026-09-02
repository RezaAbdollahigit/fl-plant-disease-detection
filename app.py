import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from models.mobilenet import get_mobilenet
import os
import time
import numpy as np
import cv2

# ==========================================
# 🧠 EXPLAINABLE AI (Grad-CAM)
# ==========================================
class GradCAM:
    """Hooks into the last convolutional layer to extract feature maps and gradients."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # MobileNetV2's final convolutional block
        target_layer = self.model.features[-1]
        
        # Register the hooks and keep handles to be able to safely remove them later
        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def generate_heatmap(self, input_tensor, target_class):
        self.model.zero_grad()
        
        # Safely request gradients without modifying the original tensor in-place
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        output = self.model(input_tensor)
        target = output[0][target_class]
        target.backward()

        # Mathematically multiply the gradients by the feature maps
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Clone to avoid in-place modification of computation graph tensors
        activations = self.activations.clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        
        # Prevent division by zero
        max_val = torch.max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.cpu().detach().numpy()

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks during multiple UI inferences."""
        self.forward_handle.remove()
        self.backward_handle.remove()

# ==========================================
# 1. Dataset Classes
# ==========================================
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# ==========================================
# 2. Load ALL Models (Cached)
# ==========================================
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

    baseline_model = load_specific_model('baseline_model.pth')
    fedavg_model = load_specific_model('fedavg_model_round_25.pth')
    fedprox_model = load_specific_model('fedprox_model_round_25.pth')
    
    return baseline_model, fedavg_model, fedprox_model, device

baseline, fedavg, fedprox, device = load_all_models()

# ==========================================
# 3. Image Preprocessing
# ==========================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0).to(device)

# ==========================================
# 4. Standardized Inference Function
# ==========================================
def get_prediction(model, tensor, original_image=None, use_gradcam=False):
    # Latency Timer Start
    start_time = time.time()
    
    if use_gradcam:
        model.eval()
        outputs = model(tensor)
    else:
        with torch.no_grad():
            outputs = model(tensor)
            
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)
    
    # Latency Timer End
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)
    
    raw_class = CLASS_NAMES[predicted_idx.item()]
    clean_name = raw_class.replace("___", " - ").replace("__", " ").replace("_", " ")
    status = "Healthy" if "healthy" in raw_class.lower() else "Infected"
    
    # Generate the Visual Heatmap Overlay
    final_image = original_image
    if use_gradcam and original_image is not None:
        cam = GradCAM(model)
        heatmap = cam.generate_heatmap(tensor, predicted_idx.item())
        cam.remove_hooks() 
        
        heatmap = cv2.resize(heatmap, (original_image.width, original_image.height))
        heatmap = np.uint8(255 * heatmap)
        
        # OpenCV uses BGR, but PIL uses RGB
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        original_np = np.array(original_image)
        superimposed_img = cv2.addWeighted(original_np, 0.5, heatmap, 0.5, 0)
        final_image = Image.fromarray(superimposed_img)

    return clean_name, round(confidence.item() * 100, 2), status, latency_ms, final_image

# ==========================================
# 🖥️ UI LAYOUT & DASHBOARD
# ==========================================
def main():
    st.set_page_config(page_title="FL Defense Dashboard", page_icon="🌿", layout="wide")

    st.title("🌿 Edge Diagnostics: FedAvg vs. FedProx")
    st.write("Upload a single edge-device image to evaluate catastrophic forgetting in standard FedAvg and the stabilizing power of the FedProx proximal penalty.")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose a leaf image from the edge device...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col_img, col_empty = st.columns([1, 2])
        with col_img:
            st.image(image, caption="Uploaded Leaf", use_container_width=True)

        if st.button("Run Comparative Inference", type="primary"):
            tensor = preprocess_image(image)
            
            with st.spinner('Running multi-model GPU inference and generating AI Heatmaps...'):
                base_class, base_conf, base_stat, base_lat, base_img = get_prediction(baseline, tensor, image, use_gradcam=True)
                avg_class, avg_conf, avg_stat, avg_lat, avg_img = get_prediction(fedavg, tensor, image, use_gradcam=True)
                prox_class, prox_conf, prox_stat, prox_lat, prox_img = get_prediction(fedprox, tensor, image, use_gradcam=True)
            
            st.markdown("### 📊 Live Model Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**1. Centralized Baseline**\n\n*(The 'Gold Standard' with access to all data)*")
                st.image(base_img, caption="AI Focus Heatmap", use_container_width=True)
                st.metric(label="Predicted Disease", value=base_class)
                st.metric(label="Confidence", value=f"{base_conf}%")
                st.metric(label="Edge Latency", value=f"{base_lat} ms")
                
            with col2:
                st.warning("**2. Standard FedAvg**\n\n*(Suffers from Catastrophic Forgetting)*")
                st.image(avg_img, caption="AI Focus Heatmap", use_container_width=True)
                st.metric(label="Predicted Disease", value=avg_class)
                st.metric(label="Confidence", value=f"{avg_conf}%", delta="- Broken / Confused" if avg_class != base_class else "Agrees with Baseline", delta_color="inverse" if avg_class != base_class else "normal")
                st.metric(label="Edge Latency", value=f"{avg_lat} ms")
                
            with col3:
                st.success("**3. FedProx (Proposed)**\n\n*(Stabilized via Proximal Penalty)*")
                st.image(prox_img, caption="AI Focus Heatmap", use_container_width=True)
                st.metric(label="Predicted Disease", value=prox_class)
                st.metric(label="Confidence", value=f"{prox_conf}%", delta="Corrected Alignment" if prox_class == base_class and avg_class != base_class else None)
                st.metric(label="Edge Latency", value=f"{prox_lat} ms")

if __name__ == "__main__":
    main()