import torch
import torch.nn as nn
import h5py
import numpy as np
from tqdm import tqdm
from se3_transformer_pytorch import SE3Transformer

class SE3ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(SE3ClassificationModel, self).__init__()
        self.feature_proj = nn.Linear(1, 8)  # Smaller feature projection
        self.se3_transformer = SE3Transformer(
            dim=8,       # Minimal feature dimension
            depth=1,     # Single transformer layer
            heads=1,     # Single attention head
            num_degrees=1 # Minimal degrees of equivariance
        )
        self.fc = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, features, coords):
        features = features.view(-1, 1)
        features = self.feature_proj(features)
        features = features.view(coords.shape[0], coords.shape[1], -1)
        transformed = torch.utils.checkpoint.checkpoint(self.se3_transformer, features, coords)
        pooled = torch.max(transformed, dim=1)[0]
        return self.fc(pooled)
    
# Load test data from HDF5
def load_test_data(test_h5_file):
    with h5py.File(test_h5_file, 'r') as f:
        coords = torch.tensor(f["coords"][:], dtype=torch.float32)
        features = torch.tensor(f["features"][:], dtype=torch.float32)
        labels = torch.tensor(f["labels"][:], dtype=torch.long)  # Assuming labels are integers
    return coords, features, labels

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SE3ClassificationModel(num_classes=3)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

print("okay here")

# Load your test dataset
test_h5_file = "datatest_150i_1024.h5"
test_coords, test_features, test_labels = load_test_data(test_h5_file)

# Move data to the appropriate device
test_coords = test_coords.to(device)
test_features = test_features.to(device)
test_labels = test_labels.to(device)

# Run inference on the test dataset
correct = 0
total = 0

with torch.no_grad():
    for i in tqdm(range(total)):
        coords = test_coords[i].unsqueeze(0).to(device)  # Add batch dimension
        features = test_features[i].unsqueeze(0).to(device)  # Add batch dimension
        outputs = model(features, coords)
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == test_labels[i].to(device)).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")