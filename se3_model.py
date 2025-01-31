import torch
import torch.nn as nn
import torch.nn.functional as F
from se3_transformer_pytorch import SE3Transformer
from tqdm import tqdm
import h5py
import numpy as np
from torch.amp import GradScaler, autocast


# SE3 Classification Model
class SE3ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(SE3ClassificationModel, self).__init__()
        self.feature_proj = nn.Linear(1, 32)  # Reduced feature dimension
        self.se3_transformer = SE3Transformer(
            dim=32,  # Reduced input feature dimension
            depth=4,  # Reduced number of layers
            heads=4,  # Reduced attention heads
            num_degrees=3  # Reduced degrees for equivariance
        )
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, features, coords):
        # Reshape features to be compatible with Linear layer
        features = features.view(-1, 1)  # Flatten to [batch_size * num_points, feature_dim]

        # Project features to dimension 32
        features = self.feature_proj(features)

        # Reshape back to [batch_size, num_points, feature_dim] for SE3Transformer
        features = features.view(coords.shape[0], coords.shape[1], -1)

        # Pass through SE3 Transformer with gradient checkpointing
        transformed = torch.utils.checkpoint.checkpoint(self.se3_transformer, features, coords)

        # Global max pooling
        pooled = torch.max(transformed, dim=1)[0]

        # Fully connected classification head
        return self.fc(pooled)


def pad_point_clouds(coords_list, features_list, labels_list, max_points):
    """
    Pad point clouds, features, and labels to ensure uniform size across the dataset.
    
    Args:
        coords_list: List of coordinates arrays.
        features_list: List of features arrays.
        labels_list: List of labels (one label per sample).
        max_points: Maximum number of points to pad/truncate to.

    Returns:
        Padded coordinates, features, and labels as tensors.
    """
    padded_coords, padded_features, padded_labels = [], [], []

    for coords, features, label in zip(coords_list, features_list, labels_list):
        num_points = coords.shape[0]

        if num_points > max_points:
            # Truncate if the number of points exceeds max_points
            coords = coords[:max_points]
            features = features[:max_points]
        else:
            # Pad if the number of points is less than max_points
            padding = max_points - num_points
            coords = F.pad(coords, (0, 0, 0, padding), "constant", 0)
            features = F.pad(features, (0, 0, 0, padding), "constant", 0)

        # Ensure the label is a single integer
        if isinstance(label, torch.Tensor):
            label = label.item() if label.numel() == 1 else label[0].item()
        elif isinstance(label, np.ndarray):
            label = int(label[0]) if label.size == 1 else int(label.flat[0])
        else:
            label = int(label)

        # Add to the padded lists
        padded_coords.append(coords)
        padded_features.append(features)
        padded_labels.append(label)

    return (
        torch.stack(padded_coords),  # Stack coords into a single tensor
        torch.stack(padded_features),  # Stack features into a single tensor
        torch.tensor(padded_labels, dtype=torch.long),  # Convert labels to tensor
    )


def load_data_from_h5(h5_file):
    with h5py.File(h5_file, "r") as f:
        coords_list, features_list, labels_list = [], [], []
        for label_name in f.keys():
            coords = torch.tensor(f[label_name]["coords"][:], dtype=torch.float32)
            features = torch.tensor(f[label_name]["features"][:], dtype=torch.float32)
            labels = torch.tensor(f[label_name]["labels"][:], dtype=torch.long)
            coords_list.append(coords)
            features_list.append(features)
            labels_list.append(labels)
        return coords_list, features_list, labels_list


def check_memory(stage=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{stage}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def train_model(model, data_loader, optimizer, criterion, num_epochs=5):
    model.train()
    scaler = GradScaler("cuda")  # Specify 'cuda' for mixed precision
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for coords, features, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            coords, features, labels = coords.to(device), features.to(device), labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast("cuda"):
                outputs = model(features, coords)
                loss = criterion(outputs, labels)

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    # Load data from the HDF5 file
    h5_file = "revised_chunks_dataset.h5"
    coords_list, features_list, labels_list = load_data_from_h5(h5_file)

    # Limit maximum points
    max_points = 1024  # Further reduced to save memory
    print(f"Using max_points: {max_points}")

    # Pad the data
    coords, features, labels = pad_point_clouds(coords_list, features_list, labels_list, max_points)

    # Split into train and test sets
    split_idx = int(0.8 * len(labels))
    train_coords, test_coords = coords[:split_idx], coords[split_idx:]
    train_features, test_features = features[:split_idx], features[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]

    # Create DataLoader for training
    train_dataset = torch.utils.data.TensorDataset(train_coords, train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Initialize the model, optimizer, and loss function
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SE3ClassificationModel(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Monitor memory before training
    print("Before Training:")
    check_memory("Before Training")

    # Train the model
    train_model(model, train_loader, optimizer, criterion, num_epochs=5)

    # Monitor memory after training
    print("After Training:")
    check_memory("After Training")
