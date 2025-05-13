import torch
from typing import Sequence, Union
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class similarityScores:

    def __init__(self, activations: Sequence[str]):
        """
        input: Possibly nested sequence of strings of experiment names

        This will build: 
        - self.activations_dict: { key: tensor } where `key` is a tuple of indices
        - self.name_to_indices: { name: indices } for reverse lookup by experiment name
        """
        self.activations_dict = {}
        self.name_to_indices = {}  
        self._flatten_activations(activations, path=())
    
    def _flatten_activations(self, activations, path):
        if isinstance(activations, str):
            self.activations_dict[path] = torch.load(f'runs/{activations}/analysis/output_second_last_linear_cifar10_test.pt')
            self.name_to_indices[activations] = path

        elif hasattr(activations, '__iter__') and not isinstance(activations, (np.float32, np.float64, float)):
            for i, activation in enumerate(activations):
                self._flatten_activations(activation, path + (i,))

    def get_activations(self, identifier):
        """
        Get activations by either indices tuple or experiment name
        Args:
            identifier: Either a tuple of indices or a string experiment name  
        Returns:
            torch.Tensor: The activation tensor
        """
        if isinstance(identifier, tuple):
            return self.activations_dict[identifier]  # Now returning tensor directly, not [0]
        elif isinstance(identifier, str):
            indices = self.name_to_indices[identifier]
            return self.activations_dict[indices]  # Now returning tensor directly, not [0]
        else:
            raise ValueError("identifier must be either a tuple of indices or a string experiment name")
        
    def dist_matrix(self, identifier, metric: str = 'euclidean', num_samples: int = 1000) -> torch.Tensor:
        """
        computes distance matrix for a single set of activations

        Args:
            identifier: Either a tuple of indices or a string experiment name
            metric: one of 'euclidean', 'cosine', or 'correlation'
            
        Returns:
            D: Tensor of shape (n, n) with D[i,j] = distance between x[i] and x[j]
        """
        
        # Subsample to first num_samples points
        activation = self.get_activations(identifier)
        if activation.shape[0] > num_samples:
            activation = activation[:num_samples]

        if metric == 'euclidean':
            return torch.cdist(activation, activation, p=2)
        
        elif metric == 'cosine':
            activation_norm = F.normalize(activation, p=2, dim=1)         
            sim = activation_norm @ activation_norm.t()                    
            return 1 - sim
        
        elif metric == 'correlation':
            x_centered = activation - activation.mean(dim=1, keepdim=True) 
            x_normed = F.normalize(x_centered, p=2, dim=1)
            corr = x_normed @ x_normed.t()
            return 1 - corr
        else:
            raise ValueError(f"Unknown metric '{metric}'")

    def mknn(self, identifier1, identifier2, k, metric: str = 'euclidean', num_samples: int = 1000) -> torch.Tensor:
        """
        Implements mkNN similarity score.

        Args:
            identifier1: Either a tuple of indices or a string experiment name
            identifier2: Either a tuple of indices or a string experiment name
            k: number of nearest neighbors to consider
            metric: one of 'euclidean', 'cosine', or 'correlation'

        Returns: 
            float: similarity score between the two activations
        """
        activations1 = self.get_activations(identifier1)
        activations2 = self.get_activations(identifier2)
        dist_matrix_1 = self.dist_matrix(identifier1, metric)
        dist_matrix_2 = self.dist_matrix(identifier2, metric)

        topk_distances_exp1, topk_indices_exp1 = torch.topk(dist_matrix_1, k + 1, dim=1, largest=False)
        topk_distances_exp2, topk_indices_exp2 = torch.topk(dist_matrix_2, k + 1, dim=1, largest=False)

        nearest_neighbor_indices_exp1 = topk_indices_exp1[:, 1:]
        nearest_neighbor_indices_exp2 = topk_indices_exp2[:, 1:]

        num_rows = nearest_neighbor_indices_exp1.shape[0]
        common_neighbors_counts = torch.zeros(num_rows, dtype=torch.long)
        for r_idx in range(num_rows):
            set1 = set(nearest_neighbor_indices_exp1[r_idx].tolist())
            set2 = set(nearest_neighbor_indices_exp2[r_idx].tolist())
            common_neighbors_counts[r_idx] = len(set1.intersection(set2))
        
        return torch.mean(common_neighbors_counts / k)

    def mknn_experiment(self, pairs, write_path, metric = 'euclidean'):
        """
        Compute MkNN similarity for multiple pairs of experiments and create a plot.
        
        Args:
            pairs: List of pairs of experiment names
            write_path: Path to save the plot
            metric: Distance metric to use
        """
        k_values = np.arange(1, 100, 5)
        results = np.zeros((len(pairs), len(k_values)))

        # Compute MkNN values for each pair and k
        for idx, k in enumerate(k_values):
            print(f"Computing MkNN for k={k}")
            for i, pair in enumerate(pairs):
                results[i, idx] = self.mknn(pair[0], pair[1], k, metric)
                
        # Create the plot
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")

        # Plot each pair's results
        for i, pair in enumerate(pairs):
            label = f"{pair[0]} / {pair[1]}"
            plt.plot(k_values, results[i, :], marker='o', linewidth=2, label=label)

        plt.title('MkNN Similarity vs. k for Experiment Pairs', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Mean k-Nearest Neighbor Overlap (MkNN)', fontsize=14)
        
        plt.legend(title='Experiment Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0, 0, 0.85, 1])  
        plt.savefig(write_path)
        print(f"Plot saved to {write_path}")
        plt.close()
    
    def jaccard(self, identifier1, identifier2, k, metric: str = 'euclidean', num_samples: int = 1000) -> torch.Tensor:
        """
        Implements neighborhood Jaccard similarity score.

        Args:
            identifier1: Either a tuple of indices or a string experiment name
            identifier2: Either a tuple of indices or a string experiment name
            k: number of nearest neighbors to consider
            metric: one of 'euclidean', 'cosine', or 'correlation'

        Returns: 
            float: mean Jaccard similarity between the two activations
        """
        # compute distance matrices
        dm1 = self.dist_matrix(identifier1, metric, num_samples)
        dm2 = self.dist_matrix(identifier2, metric, num_samples)

        # get top-k+1 (including self) indices, then drop self
        topk1 = torch.topk(dm1, k + 1, dim=1, largest=False)[1][:, 1:]
        topk2 = torch.topk(dm2, k + 1, dim=1, largest=False)[1][:, 1:]

        n = topk1.shape[0]
        jaccard_scores = torch.zeros(n)
        for i in range(n):
            set1 = set(topk1[i].tolist())
            set2 = set(topk2[i].tolist())
            union = set1 | set2
            if union:
                jaccard_scores[i] = len(set1 & set2) / len(union)
            else:
                jaccard_scores[i] = 0.0
        return jaccard_scores.mean()

    def jaccard_experiment(self, pairs, write_path, metric='euclidean'):
        """
        Compute Jaccard similarity for multiple pairs of experiments and create a plot.
        """
        k_values = np.arange(1, 100, 5)
        results = np.zeros((len(pairs), len(k_values)))

        for idx, k in enumerate(k_values):
            print(f"Computing Jaccard for k={k}")
            for i, pair in enumerate(pairs):
                results[i, idx] = self.jaccard(pair[0], pair[1], k, metric)
                
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")

        for i, pair in enumerate(pairs):
            label = f"{pair[0]} / {pair[1]}"
            plt.plot(k_values, results[i, :], marker='o', linewidth=2, label=label)

        plt.title('Jaccard Similarity vs. k for Experiment Pairs', fontsize=16)
        plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
        plt.ylabel('Mean Neighborhood Jaccard', fontsize=14)
        plt.legend(title='Experiment Pair', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(write_path)
        print(f"Plot saved to {write_path}")
        plt.close()
    

if __name__ == "__main__":

    activation_tensors = [('mlp_5_256_adam', 'mlp_5_256_adam3', 'mlp_5_256_adam2'), ('mlp_5_256_muon', 'mlp_5_256_muon2', 'mlp_5_256_muon3')]

    similarity_scores = similarityScores(activation_tensors)

    metric = 'euclidean'
    pairs = [['mlp_5_256_adam', 'mlp_5_256_adam3'], ['mlp_5_256_adam2', 'mlp_5_256_adam3'], ['mlp_5_256_adam', 'mlp_5_256_adam2'], ['mlp_5_256_muon', 'mlp_5_256_muon2'], ['mlp_5_256_muon2', 'mlp_5_256_muon3'], ['mlp_5_256_muon', 'mlp_5_256_muon3']]
    
    similarity_scores.mknn_experiment(pairs, f'analysis/results/mknn_{metric}.png', metric = metric)

