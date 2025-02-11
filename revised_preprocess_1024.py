import os
import numpy as np
import h5py
import nibabel as nib  # For loading NIfTI images
from tqdm import tqdm

def volumetric_to_point_cloud(volume):
    points = np.argwhere(volume > 0)
    features = volume[volume > 0].astype(np.float32)[:, None]  # Extract features (intensities) of non-zero voxels
    return points, features

class_labels = {
    'AVM': 0,
    'Pituitary': 1,
    'Schwannoma': 2
}

def process_and_save_data(input_folder, output_h5_file, num_points=1024):
    
    coords_list = []
    features_list = []
    labels_list = []

    # Iterate over each class directory
    for class_name in os.listdir(input_folder):
        class_dir = os.path.join(input_folder, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                    # Load the NIfTI image
                    image_path = os.path.join(class_dir, filename)
                    volume = nib.load(image_path).get_fdata()  # Load the volume data

                    # Convert volume to point cloud
                    points, features = volumetric_to_point_cloud(volume)

                    # Limit to num_points
                    if len(points) > num_points:
                        indices = np.random.choice(len(points), num_points, replace=False)
                        points = points[indices]
                        features = features[indices]

                    # Store coordinates, features, and labels
                    coords_list.append(points)
                    features_list.append(features)
                    labels = np.full(points.shape[0], fill_value=class_labels[class_name])  # Use the mapping for labels
                    labels_list.append(labels)

    # Convert lists to numpy arrays
    coords_array = np.vstack(coords_list)  # Stack arrays vertically
    features_array = np.vstack(features_list)  # Stack arrays vertically
    labels_array = np.concatenate(labels_list)  # Concatenate labels into a single array

    # Save to HDF5
    with h5py.File(output_h5_file, 'w') as h5f:
        h5f.create_dataset('coords', data=coords_array, compression='gzip')
        h5f.create_dataset('features', data=features_array, compression='gzip')
        h5f.create_dataset('labels', data=labels_array, compression='gzip')

if __name__ == "__main__":
    num_points = 1024  # Define the number of points
    input_folder = 'classification_images_200/test_dataset/image/'  # Path to your dataset folder
    output_h5_file = 'datatest_1024.h5'  # Output HDF5 file
    process_and_save_data(input_folder, output_h5_file, num_points)