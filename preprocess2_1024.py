import os
import numpy as np
import h5py
from tqdm import tqdm
import nibabel as nib  # For loading NIfTI images

def volumetric_to_point_cloud(volume):
    """
    Convert a 3D volume into a point cloud with features and coordinates.
    """
    # Get indices of non-zero voxels (points)
    points = np.argwhere(volume > 0)  # Shape: [num_points, 3]
    # Normalize coordinates to range [0, 1]
    points = points / np.array(volume.shape)  # Normalize each dimension
    return points

def process_images_to_hdf5(input_folder, output_h5_file):
    # Create a mapping from class names to integer labels
    class_mapping = {name: idx for idx, name in enumerate(os.listdir(input_folder))}
    
    coords_list = []
    features_list = []
    labels_list = []

    with h5py.File(output_h5_file, 'w') as h5f:
        for label_name in class_mapping.keys():
            class_folder = os.path.join(input_folder, label_name)
            for filename in tqdm(os.listdir(class_folder), desc=f"Processing {label_name}"):
                if not filename.endswith('.nii'):
                    continue  # Skip non-NIfTI files

                # Load the NIfTI image
                file_path = os.path.join(class_folder, filename)
                volume = nib.load(file_path).get_fdata()  # Load the volume data

                # Convert volume to point cloud
                points = volumetric_to_point_cloud(volume)

                # Store coordinates, features, and labels
                coords_list.append(points)  # Store the coordinates
                features_list.append(volume[volume > 0])  # Example: using non-zero voxel values as features
                labels = np.full(points.shape[0], fill_value=class_mapping[label_name])  # Use the mapping for labels
                labels_list.append(labels)

        # Save to HDF5
        h5f.create_dataset('coords', data=np.array(coords_list, dtype=object))  # Store as object for variable-length arrays
        h5f.create_dataset('features', data=np.array(features_list, dtype=object))  # Store as object for variable-length arrays
        h5f.create_dataset('labels', data=np.array(labels_list, dtype=object))  # Store as object for variable-length arrays

# Usage
input_folder = 'classifiication_images_200/test_dataset/image'  # Path to your dataset
output_h5_file = 'test_dataset.h5'  # Output HDF5 file
process_images_to_hdf5(input_folder, output_h5_file)
