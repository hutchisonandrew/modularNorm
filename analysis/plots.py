import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Read the CSV file
csv_path = 'runs/mlp_2_512_spectral/metrics/perturbation_analysis/perturbation_analysis.csv'
df = pd.read_csv(csv_path)

experiment_name = 'experiment1_spectral'
epoch = 5
# Filter data for epoch 0 only
epoch_df = df[df['epoch'] == epoch]

# Create output directory for plots if it doesn't exist
output_dir = f'analysis/{experiment_name}/plots/epoch_{epoch}'
os.makedirs(output_dir, exist_ok=True)

# Group by magnitude_iteration and create plots
for magnitude_iter in range(10):  # 0 to 9
    # Filter data for this magnitude iteration
    iter_data = epoch_df[epoch_df['magnitude_iteration'] == magnitude_iter]
    print(len(iter_data))
    
    if iter_data.empty:
        print(f"No data found for magnitude iteration {magnitude_iter}")
        continue
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both residuals
    plt.plot(range(len(iter_data)), iter_data['frobenius_residual'], 'b-', label='Frobenius Residual')
    plt.plot(range(len(iter_data)), iter_data['spectral_residual'], 'r-', label='Spectral Residual')
    
    # Set labels and title
    plt.xlabel('Datapoint Index')
    plt.ylabel('Residual Value')
    plt.title(f'Residuals for Magnitude Iteration {magnitude_iter}')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'{output_dir}/magnitude_iter_{magnitude_iter}_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"Created plots in {output_dir}")



