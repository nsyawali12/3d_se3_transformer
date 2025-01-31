import torch
import torch.nn as nn
import torch.nn.functional as F
from se3_transformer_pytorch import SE3Transformer

# SE3 Classification Model
class OptimizedSE3ClassificationModel(nn.Module):
    def __init__(self, num_classes, feature_dim=16):
        super(OptimizedSE3ClassificationModel, self).__init__()
        self.se3_transformer = SE3Transformer(
            dim=feature_dim,  # Lower feature dimensions for efficiency
            depth=2,          # Fewer transformer layers
            heads=2,          # Fewer attention heads
            num_degrees=1,    # Lower degrees for SE(3)-equivariance
            output_degrees=1  # Degree of output representation
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, features, coords):
        # Pass through SE3 Transformer
        x = self.se3_transformer(features, coords)  # Input features and coordinates
        x = torch.max(x, dim=1)[0]  # Global max pooling across points
        x = self.fc(x)              # Fully connected layers for classification
        return x

# Dummy Data (Optimized)
batch_size = 4       # Reduce batch size
num_points = 100     # Reduce number of points per sample
feature_dim = 16     # Reduce feature dimensions
num_classes = 3      # Number of output classes

# Generate random input data
features = torch.randn(batch_size, num_points, feature_dim)  # Node features
coords = torch.randn(batch_size, num_points, 3)  # Node coordinates (x, y, z)
labels = torch.randint(0, num_classes, (batch_size,))  # Random labels for classification

# Print maximum points per sample
max_points_per_sample = features.shape[1]
print(f"Maximum points per sample: {max_points_per_sample}")

# Initialize Model, Optimizer, and Loss
model = OptimizedSE3ClassificationModel(num_classes=num_classes, feature_dim=feature_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training Loop
model.train()
for epoch in range(5):  # Train for 5 epochs
    optimizer.zero_grad()
    outputs = model(features, coords)  # Forward pass
    loss = criterion(outputs, labels)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(features, coords)  # Forward pass
    predicted = torch.argmax(outputs, dim=-1)  # Predicted classes
    accuracy = (predicted == labels).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
