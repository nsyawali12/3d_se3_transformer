import os
import nibabel as nib
import numpy as np
import h5py
from tqdm import tqdm
import random

def volumetric_to_point_cloud(volume):
    """
    Convert a 3D volume into a point cloud with features and coordinates.
    """
    # Get indices of non-zero voxels (points)
    points = np.argwhere(volume > 0)  # Shape: [num_points, 3]
    # Normalize coordinates to range [0, 1]
    points = points / np.array(volume.shape)  # Normalize each dimension
    # Extract features (intensities) of non-zero voxels
    features = volume[volume > 0].astype(np.float32)[:, None]  # Shape: [num_points, 1]
    return points, features

def downsample_point_cloud(coords, features, max_points):
    """
    Downsample a point cloud to a maximum number of points.

    Args:
        coords (np.ndarray): Array of point coordinates [num_points, 3].
        features (np.ndarray): Array of features corresponding to points [num_points, 1].
        max_points (int): Maximum number of points to retain.

    Returns:
        np.ndarray, np.ndarray: Downsampled coordinates and features.
    """
    num_points = coords.shape[0]
    if num_points > max_points:
        # Randomly select max_points indices
        selected_indices = np.random.choice(num_points, max_points, replace=False)
        coords = coords[selected_indices]
        features = features[selected_indices]
    return coords, features

def process_and_save_to_h5(image_folder, output_file, max_points=8192):
    """
    Process NII images and save point cloud data (features and coordinates) to an HDF5 file.

    Args:
        image_folder (str): Path to the folder containing labeled image subfolders.
        output_file (str): Path to the output HDF5 file.
        max_points (int): Maximum number of points to retain per sample.
    """
    label_map = {
        'AVM_img': 0,
        'Pituitary_img': 1,
        'Schwannoma_img': 2
    }

    with h5py.File(output_file, 'w') as h5f:
        for label_name, label_value in tqdm(label_map.items(), desc="Processing folders"):
            folder_path = os.path.join(image_folder, label_name)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder {folder_path} not found. Skipping.")
                continue

            group = h5f.create_group(label_name)
            features_list = []
            coords_list = []
            labels_list = []

            for filename in tqdm(os.listdir(folder_path), desc=f"Processing {label_name}"):
                if not filename.endswith('.nii'):
                    continue

                file_path = os.path.join(folder_path, filename)
                try:
                    # Load NII image
                    nii_image = nib.load(file_path)
                    volume = nii_image.get_fdata()

                    # Convert to point cloud
                    coords, features = volumetric_to_point_cloud(volume)

                    # Downsample the point cloud
                    coords, features = downsample_point_cloud(coords, features, max_points)

                    # Append data
                    coords_list.append(coords)
                    features_list.append(features)
                    labels_list.append(np.full((coords.shape[0],), label_value, dtype=np.int32))

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            # Concatenate all data for this label
            all_coords = np.concatenate(coords_list, axis=0)
            all_features = np.concatenate(features_list, axis=0)
            all_labels = np.concatenate(labels_list, axis=0)

            # Save to HDF5
            group.create_dataset('coords', data=all_coords, compression='gzip')
            group.create_dataset('features', data=all_features, compression='gzip')
            group.create_dataset('labels', data=all_labels, compression='gzip')

if __name__ == "__main__":
    image_folder = "image_chunks"  # Path to folder containing labeled subfolders
    output_file = "chunks_dataset_8192.h5"  # Path to save the HDF5 file
    max_points = 8192  # Define the maximum number of points per sample

    process_and_save_to_h5(image_folder, output_file, max_points)