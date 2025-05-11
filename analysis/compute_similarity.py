import torch
import csv
import os

# Normalize rows to unit Euclidean norm

def compute_MkNN(exp1_activations, exp2_activations, k):
    exp1_activations = exp1_activations / torch.linalg.norm(exp1_activations, ord=2, dim=1, keepdim=True)
    exp2_activations = exp2_activations / torch.linalg.norm(exp2_activations, ord=2, dim=1, keepdim=True)
    # Calculate pairwise distances for exp1
    pairwise_distances_exp1 = torch.cdist(exp1_activations, exp1_activations)
    
    # Find the k+1 nearest neighbors (including the point itself)
    # We use k+1 because the point itself will be the closest
    topk_distances_exp1, topk_indices_exp1 = torch.topk(pairwise_distances_exp1, k + 1, dim=1, largest=False)

    # Exclude the first column of indices, which corresponds to the point itself
    nearest_neighbor_indices_exp1 = topk_indices_exp1[:, 1:]

    # Now do the same for exp2_activations
    # Calculate pairwise distances for exp2
    pairwise_distances_exp2 = torch.cdist(exp2_activations, exp2_activations)


    # Find the k+1 nearest neighbors for exp2
    topk_distances_exp2, topk_indices_exp2 = torch.topk(pairwise_distances_exp2, k + 1, dim=1, largest=False)

    # Exclude the first column of indices for exp2
    nearest_neighbor_indices_exp2 = topk_indices_exp2[:, 1:]

    
    # Calculate number of common neighbors for each row
    num_rows = nearest_neighbor_indices_exp1.shape[0]
    common_neighbors_counts = torch.zeros(num_rows, dtype=torch.long)
    for r_idx in range(num_rows):
        # Convert rows to sets for efficient intersection
        set1 = set(nearest_neighbor_indices_exp1[r_idx].tolist())
        set2 = set(nearest_neighbor_indices_exp2[r_idx].tolist())
        common_neighbors_counts[r_idx] = len(set1.intersection(set2))
    
    final_MkNN = torch.mean(common_neighbors_counts / k)
    
    return final_MkNN



def save_similarity_results(exp1_name, exp2_name, k, final_MkNN_value, csv_filepath):
    # Check if the file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_filepath)
    
    # Open the CSV file in append mode. If it doesn't exist, it will be created.
    with open(csv_filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file did not exist (or is empty), write the header
        if not file_exists or os.path.getsize(csv_filepath) == 0:
            writer.writerow(["exp1_name", "exp2_name", "k", "final_MkNN"])
        
        # Write the data row
        # Convert final_MkNN from tensor to Python number if it's a tensor
        if isinstance(final_MkNN_value, torch.Tensor):
            final_MkNN_item = final_MkNN_value.item()
        else:
            final_MkNN_item = final_MkNN_value
            
        writer.writerow([exp1_name, exp2_name, k, final_MkNN_item])
 

if __name__ == "__main__":
    exp1_name = 'mlp_5_256_adam'
    exp2_name = 'mlp_5_256_adam2'

    exp1_dir = f'./runs/{exp1_name}'
    exp2_dir = f'./runs/{exp2_name}'

    csv_filepath = "analysis/similarity_results.csv"
    
    exp1_activations = torch.load(f'{exp1_dir}/analysis/output_second_last_linear_cifar10_test.pt')
    exp2_activations = torch.load(f'{exp2_dir}/analysis/output_second_last_linear_cifar10_test.pt')

    # Determine the number of rows and the number of samples to take
    num_total_rows = exp1_activations.shape[0]
    num_samples = 1000

    if num_total_rows > num_samples:
        # Select the first num_samples rows
        exp1_activations = exp1_activations[:num_samples]
        exp2_activations = exp2_activations[:num_samples]

        print(f"Selected first {exp1_activations.shape[0]} rows from the original {num_total_rows} rows.")
    else:
        print(f"Number of rows ({num_total_rows}) is not greater than num_samples ({num_samples}). Using all rows.")
    
    for k_value in range(1, 100):
        mknn_result = compute_MkNN(exp1_activations, exp2_activations, k_value)
        save_similarity_results(exp1_name, exp2_name, k_value, mknn_result, csv_filepath)