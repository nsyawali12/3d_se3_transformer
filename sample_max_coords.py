import h5py
import torch

# Function to load data from HDF5 file
def load_data_from_h5(h5_file):
    with h5py.File(h5_file, "r") as f:
        coords_list = []
        for label_name in f.keys():
            coords = torch.tensor(f[label_name]["coords"][:], dtype=torch.float32)
            coords_list.append(coords)
        return coords_list
    
# Main execution
if __name__ == "__main__":
    h5_file = "dataset_150i_1024.h5"  # Replace with the path to your dataset

    # Load coordinates from the dataset
    coords_list = load_data_from_h5(h5_file)

    # Find the maximum number of points in the dataset
    max_points = max(coords.shape[0] for coords in coords_list)

    # Print the maximum number of points
    print(f"Maximum points in a single sample: {max_points}")