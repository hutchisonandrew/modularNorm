import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_mknn_plot(csv_filepath, output_plot_path):
    """
    Generates and saves a plot of MkNN vs. k from similarity results.

    Args:
        csv_filepath (str): Path to the input CSV file.
        output_plot_path (str): Path to save the generated plot.
    """
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return

    # Create a combined column for hue to represent experiment pairs
    df['experiment_pair'] = df['exp1_name'] + ' / ' + df['exp2_name']

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid") # Using a seaborn theme

    # Create the line plot
    lineplot = sns.lineplot(
        data=df,
        x='k',
        y='final_MkNN',
        hue='experiment_pair',
        marker='o', # Add markers to points
        linewidth=2
    )

    plt.title('MkNN Similarity vs. k for Experiment Pairs', fontsize=16)
    plt.xlabel('k (Number of Nearest Neighbors)', fontsize=14)
    plt.ylabel('Mean k-Nearest Neighbor Overlap (MkNN)', fontsize=14)
    
    # Improve legend
    plt.legend(title='Experiment Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend if it's outside

    # Save the plot
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
    plt.close() # Close the plot figure

if __name__ == '__main__':
    # Assuming similarity_results.csv is in the same directory as plots.py (i.e., 'analysis/')
    csv_file = 'analysis/similarity_results.csv'
    plot_output_file = 'analysis/mknn_similarity_by_k.png'
    
    create_mknn_plot(csv_file, plot_output_file)





