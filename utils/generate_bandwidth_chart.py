import matplotlib.pyplot as plt
import os

def generate_bandwidth_chart():
    # 1. Define the Network Math
    # Centralized: Uploading raw images
    total_images = 16000
    avg_image_size_mb = 1.5
    centralized_bandwidth = total_images * avg_image_size_mb  # ~24,000 MB

    # Federated: Transmitting model weights (MobileNetV2 = ~13.6 MB)
    model_size_mb = 13.6
    num_clients = 5
    num_rounds = 25
    # (Download from server + Upload to server) * Clients * Rounds
    federated_bandwidth = (model_size_mb * 2) * num_clients * num_rounds  # ~3,400 MB

    # 2. Setup the Plot
    categories = ['Centralized Baseline\n(Raw Image Upload)', 'Federated Learning\n(Model Weight Transfer)']
    bandwidths = [centralized_bandwidth, federated_bandwidth]
    colors = ['#1f77b4', '#5cb85c'] # Blue for Centralized, Green for Federated

    plt.figure(figsize=(9, 6))
    bars = plt.bar(categories, bandwidths, color=colors, width=0.5, edgecolor='black')

    # 3. Add Data Labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        # Convert MB to GB for cleaner reading
        gb_val = yval / 1000
        plt.text(bar.get_x() + bar.get_width()/2, yval + 500, f"{gb_val:.1f} GB", 
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 4. Titles and Formatting
    plt.title('Network Bandwidth Consumption\n(Centralized vs. Federated Approach)', fontsize=14, pad=20)
    plt.ylabel('Total Data Transmitted (Megabytes)', fontsize=12)
    plt.ylim(0, 28000) # Give headroom for the text labels
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Save the Chart
    os.makedirs('results', exist_ok=True)
    save_path = 'results/network_bandwidth_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Bandwidth analytics chart generated and saved to: {save_path}")

if __name__ == "__main__":
    generate_bandwidth_chart()