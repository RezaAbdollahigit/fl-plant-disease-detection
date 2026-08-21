## 🌿 Federated Learning for Plant Disease Detection

A robust, distributed machine learning architecture designed to classify 15 categories of plant diseases using edge-device simulation. This project evaluates the performance of standard FedAvg against the FedProx proximal penalty algorithm under simulated hardware failures and severe non-IID data distributions (Dirichlet $\alpha = 0.1$).

### 🚀 Core Architecture & Features

*   **Federated Simulation Engine:** Orchestrates 5 virtual edge networks using the Flower framework to simulate decentralized agricultural environments.
*   **Classification Model:** Utilizes a custom-headed MobileNetV2 architecture optimized for constrained edge hardware.
*   **The "Chaos Engine":** Introduces a 20% dropped-connection straggler rate and non-IID Dirichlet data partitioning to rigorously test algorithmic fault tolerance.
*   **Explainable AI (XAI):** Implements Grad-CAM attention heatmaps via a live Streamlit dashboard to visually verify model feature extraction.
*   **Comparative Analytics:** Generates automated granular F1-score breakdowns, bandwidth consumption comparisons, and dual confusion matrices.

### 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed, then clone the repository and install the required dependencies:

```bash
git clone [https://github.com/yourusername/fl-plant-disease-detection.git](https://github.com/yourusername/fl-plant-disease-detection.git)
cd fl-plant-disease-detection
pip install -r requirements.txt

```

### 🖥️ Execution Pipeline

Execute the pipeline in the following order to reproduce the comparative analysis:

1. **Establish Centralized Baseline:**
`python train_baseline.py`
2. **Run Federated Simulations:**
`python run_simulation.py --algo fedavg --clients 5`
`python run_simulation.py --algo fedprox --clients 5`
3. **Launch the XAI Dashboard:**
`streamlit run app.py`

```