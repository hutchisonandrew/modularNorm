import json
import matplotlib.pyplot as plt

# Load the perturbation results from JSON file
with open('runs/testrun1/metrics/perturbation_analysis/perturbation_results_epoch_0.json', 'r') as f:
    perturbation_data = json.load(f)
    
layer = perturbation_data['layers.0']
frob_residuals = []
spec_residuals = []
for delta_w in layer:
    frob_residual = delta_w['batch_residuals_frobenius'][0]
    spec_residual = delta_w['batch_residuals_spectral'][0]
    frob_residuals.append(frob_residual)
    spec_residuals.append(spec_residual)

# Create an index for x-axis
indices = list(range(len(frob_residuals)))

# Plot both residuals on the same chart
plt.figure(figsize=(10, 6))
plt.plot(indices, frob_residuals, 'b-o', label='Frobenius Residual')
plt.plot(indices, spec_residuals, 'r-o', label='Spectral Residual')
plt.xlabel('Delta W')
plt.ylabel('Residual Value')
plt.title('Residuals of Perturbed Weights')
plt.legend()
plt.grid(True)
plt.show()
    


