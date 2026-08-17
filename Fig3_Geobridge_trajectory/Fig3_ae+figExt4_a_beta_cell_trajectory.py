import torch
import torch.nn as nn
import torch.nn.init as init
import random
import matplotlib
import matplotlib.pyplot as plt
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import copy
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import anndata
import time
from scipy.spatial.distance import cdist
import pandas as pd
import ot
import Get_Probability_Measures
from Dim_plot import phate_plot, pca_plot, variance_select_3d_plot, random_select_3d_plot
from mmd import compute_scalar_mmd, compute_linear_mmd
from Wasserstein_loss import get_OT_plan
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import FISTA_OT
from sklearn.decomposition import PCA
from Dynamic_results import plot_PCA
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

All_data = np.array(pd.read_csv('your/path/to/data/exp.csv', index_col=0)).T
All_label = pd.read_csv('your/path/to/data/group.csv', index_col=0)
cluster_labels = np.array(All_label['x'])

def pseudotime_plot(All_data, pseudotime, PCs=[0, 1]):
    pca = PCA(n_components=5)
    All_data_np = All_data
    # All_data_np = z.cpu().detach().numpy()
    data_pca_all = pca.fit_transform(All_data_np)

    # Plot the scatter plot

    discrete_cmap = plt.cm.viridis  # You can choose other colormaps, e.g., 'plasma', 'inferno', 'Reds'
    # Map unique label values to desired solid colors and legend labels
    # data_pca = pca.transform(data_trans.cpu().detach().numpy())

    fig, ax = plt.subplots(figsize=(8, 6))
    # Store the scatter plot object for the colorbar (only need one from the gradient plots)
    gradient_scatter = None
    weights = pseudotime

    # Plot with gradient color based on MET weights
    # 'c' argument takes an array of values for coloring
    scatter = ax.scatter(
        data_pca_all[:, PCs[0]],
        data_pca_all[:, PCs[1]],
        c=weights,  # Use the weights array for coloring
        cmap=discrete_cmap,  # Specify the colormap
        s=5
    )
    # ax.scatter(data_pca[:, PCs[0]], data_pca[:, PCs[1]], color='black', s=5)
    ax.legend(loc='best')  # Adjust legend location as needed

    # Set title and labels
    ax.set_title(f'PCA Embedding (2D) Colored by Weight')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar = fig.colorbar(scatter, ax=ax)
    # Show the plotplt.grid(True, linestyle='--', alpha=0.6) # Optional: Add grid
    plt.tight_layout()  # Adjust layout to prevent labels overlapping
    plt.show(block=True)

init_target = 'T2D'

def get_All_t(All_data, cluster_labels, init_target, lambda_reg):

    data_s = torch.from_numpy(All_data[cluster_labels != init_target]).to(device)
    data_t = torch.from_numpy(All_data[cluster_labels == init_target]).to(device)
    data_t1 = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s1 = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s1.values, data_t1.values, metric='euclidean')
    C2 = 1000000 * C / C.sum()
    mu = Get_Probability_Measures.kde_gene_expression(data_s1)
    nu = Get_Probability_Measures.kde_gene_expression(data_t1)
    KP = -FISTA_OT.fista_ot2(C2, mu, nu, lambda_reg=lambda_reg, max_iter=1000, tol=1e-6)[1]
    KP_max = KP.max()
    KP_min = KP.min()
    KP_scaled = 1 - ((KP - KP_min) / (KP_max - KP_min))
    pseudotime = np.zeros_like(cluster_labels).astype(float)
    pseudotime[cluster_labels != init_target] = KP_scaled
    pseudotime[cluster_labels == init_target] = 1

    All_t = np.zeros_like(cluster_labels).astype(float)
    labels = np.unique(cluster_labels)

    # Iterate over each index subset
    for label in labels:
        # Extract the original time labels for this subset
        indices = cluster_labels == label
        t_segment = pseudotime[indices]

        # Compute the mean time label for this subset
        mean_time = np.mean(t_segment)

        # Assign the mean value to the corresponding positions in All_t for all points in this subset
        All_t[indices] = mean_time
    All_t = All_t - All_t.min()
    return pseudotime, All_t

pseudotime, All_t = get_All_t(All_data, cluster_labels, init_target, lambda_reg=1e-1)
pseudotime_plot(All_data, All_t, PCs=[0, 1])
train_data = All_data
train_t = All_t

All_data = torch.from_numpy(All_data).to(device)
All_t = torch.from_numpy(All_t).to(device)
BATCHSIZE = 100
N_DIM = All_data.shape[1]
epoch = 500
val = -1
interval = 100


# we define a subnet for use inside an affine coupling block
def subnet_fc(dims_in, dims_out):
    # Define the subnet
    subnet = nn.Sequential(
        nn.Linear(dims_in, 1024),
        nn.ReLU(),
        nn.Linear(1024, dims_out)
    )

    # Initialize each layer in the subnet
    for layer in subnet:
        if isinstance(layer, nn.Linear):
            # Initialize weights with Xavier initialization
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            # Initialize biases to zero
            if layer.bias is not None:
                init.zeros_(layer.bias)

    return subnet

# Set the random seed
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # Ensure CUDA and cuDNN operations are deterministic whenever the same seed is used
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Initialize the random seed
set_seed(42)
# a simple chain of operations is collected by ReversibleSequential
inn = Ff.SequenceINN(N_DIM)
for k in range(6):
    inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=False)

optimizer = torch.optim.Adam(inn.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=interval, eta_min=1e-6)
inn.to(device)

def scaled_output(z):
    """
    Apply standard min-max normalization to z.

    Args:
    z (torch.Tensor): Input tensor.

    Returns:
    torch.Tensor: Normalized tensor.
    """
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]

    # Avoid division by zero
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)

    return z_norm

def stratified_sampling(train_data, train_t, batch_size):
    """
    Perform stratified sampling from each time point, using the same number of samples per time point to generate the subset for each training epoch.

    Args:
    - train_data: np.ndarray, training data with shape (num_samples, num_features).
    - train_t: np.ndarray, time-point labels for the training data with shape (num_samples,).
    - batch_size: int, total number of samples in each batch.

    Returns:
    - sampled_data: np.ndarray, data after stratified sampling with shape (batch_size, num_features).
    - sampled_t: np.ndarray, time-point labels after stratified sampling with shape (batch_size,).
    """
    unique_days = np.unique(train_t)
    num_days = len(unique_days)
    samples_per_day = batch_size // num_days

    if samples_per_day == 0:
        raise ValueError("Batch size is too small to sample from each time point.")

    sampled_data = []
    sampled_t = []
    indices = []
    for day in unique_days:
        day_indices = np.where(train_t == day)[0]
        sampled_indices = np.random.choice(day_indices, size=samples_per_day, replace=False)

        sampled_data.append(train_data[sampled_indices, :])
        sampled_t.append(train_t[sampled_indices])
        indices.append(sampled_indices)

    # Merge samples from all time points
    sampled_data = np.vstack(sampled_data)
    sampled_t = np.hstack(sampled_t)

    return torch.Tensor(sampled_data), torch.from_numpy(sampled_t), indices

def compute_pairwise_distance(tensor1, tensor2):
    return torch.cdist(tensor1, tensor2, p=2)  # Euclidean distance

def distance_loss_per_time_point(input, z, t, time_point):
    # Filter samples by time point
    mask = (t == time_point)
    input_subset = input[mask]
    z_subset = z[mask]

    # Compute pairwise distances between input and z
    input_distances = compute_pairwise_distance(input_subset, input_subset)
    z_distances = compute_pairwise_distance(z_subset, z_subset)

    # Set the diagonal to a very large value to exclude self-distances
    n = input_distances.shape[0]
    inf_mask = torch.eye(n, device=input.device).bool()  # Diagonal positions
    # Create a matrix with infinity on the diagonal
    inf_matrix = torch.eye(input_distances.size(0), device=input_distances.device)
    inf_matrix[inf_mask] = float('inf')
    input_distances = input_distances + inf_matrix
    z_distances = z_distances + inf_matrix

    # Find the two closest points in input, excluding self-distances
    min_input_distance, min_input_indices = torch.min(input_distances, dim=1)

    # Use min_input_indices to find the corresponding minimum distance in z
    min_z_distance = z_distances[torch.arange(n, device=input.device), min_input_indices]

    # Return the difference between the minimum input distance and the minimum z distance
    prop = min_input_distance / min_z_distance
    return prop.var(dim=0)/prop.mean()
    #return torch.abs(min_input_distance - min_z_distance).mean()


def total_distance_loss(input, z, t):
    # Get all unique time labels
    unique_times = torch.unique(t)

    # Compute the loss for each time point
    total_loss = 0
    for time_point in unique_times:
        total_loss += distance_loss_per_time_point(input, z, t, time_point)

    # Compute the mean loss across all time points
    return total_loss / len(unique_times)

def calculate_time_based_losses_cn(t: torch.Tensor,
                                   z: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Compute two loss terms based on the evolution of z over time t.

    1.  loss_mean: Measures the variance of the rate of change in the mean of z across different time intervals.
                 (Computes mean-rate-of-change vectors for each pair of time points, then computes the average pairwise squared Euclidean distance between these vectors.)
    2.  loss_mmd: Measures the variance of the normalized per-unit-time MMD (maximum mean discrepancy) between data distributions at different time points.
                (Computes per-unit-time MMD values for each pair of time points, normalizes them by their mean, and sums the squared deviations from 1.)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - loss_mean (torch.Tensor): Computed mean loss (scalar).
            - loss_mmd (torch.Tensor): Computed MMD loss (scalar).
    """

    # 1. Find unique time points and their corresponding indices
    t_unique, inverse_indices = torch.unique(t, sorted=True, return_inverse=True)
    num_time_points = len(t_unique)

    # 2. Compute the mean z value for each unique time point
    z_unique = torch.zeros((num_time_points, z.size(1)), device=z.device)
    for j in range(num_time_points):
        indices_for_t_j = (inverse_indices == j)
        z_unique[j] = z[indices_for_t_j].mean(dim=0)

    mean_diff_values = []   # Store mean-rate-of-change vectors

    # 3. & 4. Compute MMD and mean differences between pairs of unique time points
    for i in range(num_time_points):
        # Get the sample mean at time point i
        zmean_i = z_unique[i]
        for j in range(i + 1, num_time_points): # Only compute cases where j > i to avoid duplicates and self-comparisons

            # Get the sample mean at time point j
            zmean_j = z_unique[j]

            # Compute the time difference
            time_diff = t_unique[j] - t_unique[i]

            # Compute the mean difference (rate of change of the mean)
            mean_diff = (zmean_j - zmean_i) / (time_diff)
            mean_diff_values.append(mean_diff)

    # 5. Compute the final loss

    # Compute the mean loss (loss_mean)
    mean_diff_values = torch.stack(mean_diff_values) # Shape: (num_pairs, D), where num_pairs is the number of valid time-point pairs
    # Compute pairwise squared Euclidean distances between these mean-rate-of-change vectors
    # cdist computes Euclidean distances (p=2), so they need to be squared
    mean_matrix = torch.cdist(mean_diff_values, mean_diff_values, p=2).pow(2) # Shape: (num_pairs, num_pairs)
    # The loss is the mean of all these pairwise distances
    loss_mean = torch.mean(mean_matrix)

    return loss_mean

def calculate_ot_interpolation_mmd_loss(
    t: torch.Tensor,
    z: torch.Tensor,
    input_data: torch.Tensor, # Rename 'input' to avoid conflict with the Python built-in function
    inn_model: torch.nn.Module,
    get_ot_plan_func: callable,
    compute_scalar_mmd_func: callable,
    z_min: torch.Tensor,      # Minimum value for inverse normalization
    z_max: torch.Tensor,      # Maximum value for inverse normalization
    ot_reg: float = 1e-2,
    ot_max_iter: int = 1000,
    num_gap: int = 0
) -> torch.Tensor:
    """
    Compute the MMD loss based on optimal-transport interpolation.

    This loss measures the MMD distance between data points generated by optimal-transport interpolation between the start and end time points
    (after inverse propagation through the INN) and the actual input data points at intermediate time points.

    Args:
        t (torch.Tensor): One-dimensional tensor of time points with shape (N,).
        z (torch.Tensor): Data-point tensor corresponding to time points t (possibly the output of a model),
                          shape (N, D). The computation graph should be connected to this tensor.
        input_data (torch.Tensor): Original input data corresponding to time points t,
                                   shape (N, D). Used to obtain the real data at intermediate time points.
        inn_model (torch.nn.Module): Invertible neural network model that must implement forward(x, rev=Boolean).
        get_ot_plan_func (callable): Function for computing the optimal-transport plan.
                                      Signature: get_ot_plan(source, target, reg, max_iter) -> P
                                      Must be differentiable for end-to-end training.
        compute_scalar_mmd_func (callable): Function for computing scalar MMD.
                                            Signature: compute_mmd(samples1, samples2) -> scalar_tensor
                                            Should be differentiable.
        z_min (torch.Tensor): Minimum-value tensor used to inverse-normalize interpolated data.
        z_max (torch.Tensor): Maximum-value tensor used to inverse-normalize interpolated data.
        ot_reg (float, optional): Regularization strength for optimal transport. Defaults to 1e-2.
        ot_max_iter (int, optional): Maximum number of iterations for the optimal-transport solver. Defaults to 1000.

    Returns:
        torch.Tensor: Computed average MMD loss (scalar). Returns 0.0 if it cannot be computed, such as when there are fewer than three unique time points.
    """
    # 1. Find unique time points
    t_unique, inverse_indices = torch.unique(t, sorted=True, return_inverse=True)


    num_unique_times = len(t_unique)
    indices_range = list(range(num_unique_times))
    valid_pairs = [(i, j) for i in indices_range for j in indices_range if abs(j - i) > num_gap]
    start_time_idx, end_time_idx = random.choice(valid_pairs)
    # 2. Get data from the start and end time points
    # Assume t_unique is sorted: the minimum index is 0 and the maximum index is num_unique_times - 1
    # start_time_idx = 0
    # end_time_idx = num_unique_times - 1

    data_s = z[inverse_indices == start_time_idx] # Start data (z at t_min)
    data_t = z[inverse_indices == end_time_idx]   # End data (z at t_max)

    # 3. Compute the optimal-transport plan P
    # !!! Key step: get_ot_plan_func must be differentiable !!!
    P = get_ot_plan_func(data_s, data_t, reg=ot_reg, numItermax=ot_max_iter)

    # 4. Normalize P and compute the transported start-point data
    P_sum_rows = P.sum(axis=1, keepdims=True)
    # Prevent division by zero
    P_normalized = P / (P_sum_rows)

    # data_trans = torch.matmul(P_normalized.to(z.dtype), data_t.to(z.dtype)) # The original code used P_normalized * data_t
    # Based on the original code, P_normalized is (N_s, N_t) and data_t is (N_t, D)
    # Therefore this should be P_normalized @ data_t
    data_trans = torch.matmul(P_normalized.to(data_t.dtype), data_t) # Shape: (N_s, D)

    # 5. Prepare intermediate time points and the corresponding interpolation calculations
    t_max = t_unique[end_time_idx].to(torch.float)
    t_min = t_unique[start_time_idx].to(torch.float)
    time_span = t_max - t_min

    # Get all intermediate time points (excluding the first and last)
    # intermediate_t_indices = torch.arange(1, num_unique_times - 1)
    mask = torch.ones(num_unique_times, dtype=torch.bool, device=t_unique.device)
    mask[start_time_idx] = False
    t_values = t_unique[mask] # Values for the other time points

    org_mmd_values = []
    # 6. Iterate over each intermediate time point for calculation
    for i, t_use in enumerate(t_values):
        # Get the corresponding original input data
        # `i` is the index in t_values and corresponds to index i + 1 in t_unique
        true_data_indices = (t == t_use)
        data_true = input_data[true_data_indices]

        # Compute the interpolation ratio
        inter = (t_use - t_min) / time_span # Ensure t_use is also a float type

        # Linear interpolation (using data_s and the transported data_trans)
        data_inter_scaled = (1 - inter) * data_s + inter * data_trans

        # Inverse-normalize / inverse-scale
        scale = z_max - z_min
        data_inter = data_inter_scaled * scale + z_min

        # Obtain the corresponding "original-space" data by running the INN in reverse
        data_rev = inn_model(data_inter, rev=True)[0] # Assume inn is on z.device

        # Compute MMD between the inverse-mapped interpolated data and the real data
        org_mmd = compute_scalar_mmd_func(data_rev, data_true.to(data_rev.dtype)) # Ensure matching dtypes
        org_mmd_values.append(org_mmd)

    # 7. Compute the final loss
    loss_org_mmd_values = torch.stack(org_mmd_values)
    loss_org_mmd = loss_org_mmd_values.mean()

    return loss_org_mmd

best_model = None
# Create lists for saving training and validation losses for each epoch
org_mmd_losses = []
pseudotime_list = []
All_t_list = []
z_list = []
pseudotime_list.append(pseudotime)
All_t_list.append(All_t.cpu().detach().numpy())
min_loss = float('inf')
# a very basic training loop
start_time = time.time()
for r in range(epoch):

    optimizer.zero_grad()
    # Use stratified sampling to generate batch data
    input, t, indices = stratified_sampling(train_data, train_t, BATCHSIZE)
    input = input.to(device)
    t = t.to(device)
    # pass to INN and get transformed variable z and log Jacobian determinant
    z, _ = inn(input)
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    constraint_loss = total_distance_loss(scaled_output(input), z, t)

    loss_org_mmd = calculate_ot_interpolation_mmd_loss(
        t=t,
        z=z,
        input_data=input,
        inn_model=inn,
        get_ot_plan_func=get_OT_plan,  # Use placeholder
        compute_scalar_mmd_func=compute_linear_mmd,  # Use placeholder
        z_min=z_min,
        z_max=z_max,
        ot_reg=5e-2
    )
    loss = loss_org_mmd + 100*constraint_loss
    print(f'Epoch {r + 1}:  Loss_train = {loss}, Loss_org_mmd = {loss_org_mmd}, constraint_loss = {constraint_loss}')
    org_mmd_losses.append(loss_org_mmd.item())

    # var_losses.append(loss_var.item())
    #
    if r+1>val:
        with torch.no_grad():
            z2, _ = inn(All_data.float())
            z2 = scaled_output(z2)
            loss_val = calculate_time_based_losses_cn(torch.from_numpy(train_t).to(device), z2)
            print(loss_val)
            if loss_val < min_loss:
                min_loss = loss_val
                best_model = copy.deepcopy(inn)
                best_step = r

    if (r+1) % interval == 0:
        with torch.no_grad():
            z3 = best_model(All_data.float())[0]
            z3 = scaled_output(z3)
            All_data_np = z3.cpu().detach().numpy()
            pseudotime_z, All_t_z = get_All_t(All_data_np, cluster_labels, init_target, lambda_reg=1e-1)
            pseudotime_list.append(pseudotime_z)
            All_t_list.append(All_t_z)
            z_list.append(z3)
            train_t = All_t_z


    # backpropagate and update the weights
    loss.backward()
    torch.nn.utils.clip_grad_norm_(inn.parameters(), max_norm=1.0)  # max_norm can be adjusted
    optimizer.step()
    # Update the learning rate
    scheduler.step()

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
import sys, os
os.makedirs('your/path/to/output/beta', exist_ok=True)
os.makedirs('your/path/to/output/beta/gene_dynamic', exist_ok=True)
torch.save(best_model.state_dict(), 'your/path/to/output/beta/model.pth')

inn.load_state_dict(torch.load('your/path/to/output/beta/model.pth'))
best_model=inn
z3 = best_model(All_data.float())[0]
z3 = scaled_output(z3)
All_data_np = z3.cpu().detach().numpy()
pseudotime_z, All_t_z = get_All_t(All_data_np, cluster_labels, init_target, lambda_reg=1e-1)
All_t = All_t_z
All_t = torch.from_numpy(All_t).to(device)
pseudotime_plot(All_data.cpu().detach().numpy(), All_t.cpu().detach().numpy(), PCs=[0, 1])
pseudotime_plot(All_data.cpu().detach().numpy(), pseudotime_z, PCs=[0, 1])

labels = np.unique(cluster_labels)
records = []

for label in labels:
    indices = cluster_labels == label
    t_segment = pseudotime_z[indices]

    records.append({
        "label": label,
        "pseudotime_min": np.min(t_segment),
        "pseudotime_max": np.max(t_segment)
    })

df_pseudotime_range = pd.DataFrame(records)

from matplotlib.animation import FuncAnimation
####Draw animation
reg = 1e-1
target = (cluster_labels == 'T2D')
source = (cluster_labels == 'Normal')
method = 'kde'
PCs = [0, 1]
num_inter = 100
name = 'ebdata'
path = 'your/path/to/output/beta/dynamic_Org'
color_use = 'black'
All_label = cluster_labels
unique_labels = np.unique(All_label)
labels = unique_labels
with torch.no_grad():
    z, _ = best_model(All_data.float())
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    data_t = z[target]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[source]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s.values, data_t.values, metric='euclidean')

    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)
    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans = np.dot(P_normalized, data_t.values)
    data_trans = torch.from_numpy(data_trans).to(device).to(torch.float32)
    data_s = torch.from_numpy(data_s.values).to(device)

    data_trans = data_trans * (z_max - z_min + 1e-8) + z_min
    data_s = data_s * (z_max - z_min + 1e-8) + z_min

with torch.no_grad():
    pca = PCA(n_components=5)
    All_data_np = All_data.cpu().detach().numpy()
    data_pca_all = pca.fit_transform(All_data_np)

    # Plot the scatter plot
    fig, ax = plt.subplots()  # Create a 2D figure with plt.subplots
    colors_order = ['green', 'orange', 'red']
    for i, color in enumerate(colors_order):
        # Find the label corresponding to the current color
        current_label = labels[i]
        indices = [idx for idx, l in enumerate(All_label) if l == current_label]

        # Plot the 2D scatter plot without the z-axis
        ax.scatter(data_pca_all[indices, PCs[0]], data_pca_all[indices, PCs[1]], color=color, s=5, label=labels[i])

    # Add the legend
    ax.legend()

    # Set title and labels
    ax.set_title('PCA Embedding (2D)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    t_values = torch.linspace(0.0, 1.0, num_inter + 1)
    t_values = t_values[0:]

    # Interval for printing progress
    print_interval = max(1, num_inter // 10)  # Print once every 10%
    # Initialize two scatter objects
    sc = ax.scatter([], [], c=color_use, s=1, label=f'Path {name}')

    def init():
        """Initialize the animation background"""
        sc.set_offsets(np.empty((0, 2)))
        return sc,

    def update(frame):
        """Function for updating each frame"""
        t = t_values[frame].item()
        with torch.no_grad():
            # Compute interpolated data
            data_inter = (1 - t) * data_s + t * data_trans

            data_inter = best_model(data_inter.to(torch.float).to(device), rev=True)[0]

            # Convert to a NumPy array
            data_org_np = data_inter.cpu().detach().numpy()

            # PCA dimensionality reduction
            data_pca_org = pca.transform(data_org_np)

            # Update the scatter data
            sc.set_offsets(data_pca_org[:, PCs])


        # Print progress
        if (frame + 1) % print_interval == 0 or frame == num_inter - 1:
            print(f'[{frame + 1}/{num_inter}]')
        return sc,

    ani = FuncAnimation(
        fig,
        update,
        frames=num_inter + 1,
        init_func=init,
        blit=True,
        interval=100
    )

    ani.save(f'{path}.gif', writer='imagemagick')
    plt.show(block=True)  # This line ensures the animation finishes before subsequent code runs


def new_plot_comparisions(
        df, trajectories,
        palette='viridis',
        df_time_key='samples',
        x=0, y=1,
        groups=None
):
    if groups is None:
        groups = sorted(df[df_time_key].unique())
    cmap = plt.cm.viridis
    sns.set_palette(palette)
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, len(groups) + 1))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })

    n_cols = 1
    n_rols = 1

    grid_figsize = [12, 8]
    dpi = 80
    grid_figsize = (grid_figsize[0] * n_cols, grid_figsize[1] * n_rols)
    fig = plt.figure(None, grid_figsize, dpi=dpi)

    hspace = 0.3
    wspace = None
    gspec = plt.GridSpec(n_rols, n_cols, fig, hspace=hspace, wspace=wspace)

    outline_width = (0.3, 0.05)
    size = 250
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    # plt.legend(frameon=False)
    states = sorted(df[df_time_key].unique())
    color_map = {
        'Normal': 'green',
        'Obesity': 'gold',
        'T2D': 'red'
    }

    colors = df['labels'].map(color_map)
    axs = []
    for i, gs in enumerate(gspec):
        ax = plt.subplot(gs)

        n = 0.3
        ax.scatter(
            df[x], df[y],
            c=colors,
            s=size,
            alpha=1,
            marker='X',
            linewidths=0,
            edgecolors=None
        )

        for trajectory in trajectories:
            plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5, color='Black');
        labels = df['labels'].unique()
        legend_elements = [
            Line2D(
                [0], [0], marker='o',
                color=color_map[labels[i]], label=f'{labels[i]}',
                markerfacecolor=color_map[labels[i]], markersize=15,
            )
            for i, state in enumerate(states)
        ]

        leg = ax.legend(handles=legend_elements, loc='upper left')
        ax.add_artist(leg)

        legend_elements = [
            Line2D([0], [0], color='black', lw=2, label='Trajectory')

        ]
        leg = plt.legend(handles=legend_elements, loc='upper right')
        ax.add_artist(leg)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # Change the border color
            spine.set_linewidth(0)  # Line width is optional
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        # kwargs = dict(bottom=False, left=False, labelbottom=False, labelleft=False)
        # ax.tick_params(which="both", **kwargs)
        # ax.set_frame_on(False)
        ax.patch.set_alpha(0)

        axs.append(ax)
    return fig

def inter_plot(data_use, t_use, All_data, All_label, inn, num_inter=100, traj_num=30, method='kde', plot='org', reg=1e-2, outputdir=''):

    z, _ = inn(data_use)
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    data_t = z[t_use == t_use.max()]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[t_use == t_use.min()]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s.values, data_t.values, metric='euclidean')
    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)
    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans = np.dot(P_normalized, data_t.values)
    data_trans = torch.from_numpy(data_trans).to(device).to(torch.float32)
    data_s = torch.from_numpy(data_s.values).to(device)
    # 1. Apply PCA dimensionality reduction to All_data
    pca = PCA(n_components=50)
    if plot == 'org':
        All_data_np = All_data.cpu().detach().numpy()
    else:
        with torch.no_grad():
            z_all, _ = inn(All_data)
            z_all = (z_all - z_min) / (z_max - z_min + 1e-8)
        All_data_np = z_all.cpu().detach().numpy()
    data_pca_all = pca.fit_transform(All_data_np)
    df = pd.DataFrame(data_pca_all)
    df.insert(0, 'samples', All_t.cpu().detach().numpy())
    df.insert(0, 'labels', All_label)
    if plot == 'org':
        data_trans = data_trans * (z_max - z_min + 1e-8) + z_min
        data_s = data_s * (z_max - z_min + 1e-8) + z_min
    t_values = torch.linspace(0.0, 1.0, num_inter+1)
    t_values = t_values[0:].to(device)
    selected_indices = np.random.choice(data_s.shape[0], size=traj_num, replace=False)
    trajectories = []
    with torch.no_grad():
        for i in selected_indices:
            interpolated_data = (1 - t_values.unsqueeze(1)) * data_s[i, :] + t_values.unsqueeze(1) * data_trans[i, :]
            org_z = interpolated_data
            if plot == 'org':
                org_z = inn(interpolated_data, rev=True)[0]
            trajectory = pca.transform(org_z.cpu().detach().numpy())
            trajectories.append(trajectory)

        fig = new_plot_comparisions(df, trajectories)
        plt.savefig(f"{outputdir}/{traj_num}_{plot}.png", format='png')
        plt.close(fig)  # Close the current figure to free memory
        return df

outputdir = 'your/path/to/output/beta/'

df = inter_plot(All_data.float(), All_t, All_data.float(), cluster_labels, inn, num_inter=100, traj_num=112,  method = 'kde', plot = 'org', reg = 1e-1, outputdir=outputdir)
df = inter_plot(All_data.float(), All_t, All_data.float(), cluster_labels, inn, num_inter=100, traj_num=112,  method = 'kde', plot = 'eu', reg = 1e-1, outputdir=outputdir)

df = inter_plot(All_data.float(), All_t, All_data.float(), cluster_labels, inn, num_inter=100, traj_num=10,  method = 'kde', plot = 'org', reg = 1e-1, outputdir=outputdir)
df = inter_plot(All_data.float(), All_t, All_data.float(), cluster_labels, inn, num_inter=100, traj_num=10,  method = 'kde', plot = 'eu', reg = 1e-1, outputdir=outputdir)


import Detect_driver
source = cluster_labels == 'Obesity'
target = cluster_labels == 'T2D'
driver_index_MET, init_driver_matrix = Detect_driver.Detect_driver(All_data.float(), All_t*10, best_model, np.ones(source.sum()), source=source, target=target, method='kde', reg=1e-1, name=1)
dynamic_driver_index_matrix_MET = torch.stack(driver_index_MET).cpu().numpy()
averaged_driver_index_MET = torch.mean(torch.stack(driver_index_MET), dim=0)

sorted_results_MET = torch.sort(averaged_driver_index_MET, descending=True)
sorted_driver_scores_MET = sorted_results_MET[0] # Sorted driver_index values
original_indices_MET = sorted_results_MET[1]   # Original indices corresponding to the sorted values

HVG = pd.read_csv('your/path/to/data/exp.csv', index_col=0).T.columns
driver_genes_MET = HVG[original_indices_MET[0:100].cpu().detach().numpy()]

df = pd.DataFrame(dynamic_driver_index_matrix_MET, columns=HVG)
df.to_csv(f'{outputdir}/driver_matrix.csv')
df = pd.DataFrame(init_driver_matrix.cpu().numpy(), columns=HVG)
df.to_csv(f'{outputdir}/init_driver_matrix.csv')
# Matrix shape: time points x number of genes
MET = dynamic_driver_index_matrix_MET[:, original_indices_MET[0:100].cpu().detach().numpy()].T

# Apply row normalization to MET and EMT separately
MET_norm = Detect_driver.row_normalize(MET)
driver_genes_MET = HVG[original_indices_MET[0:100].cpu().detach().numpy()]
Detect_driver.plot_phase_heatmap(MET_norm, 'purple', 2, driver_genes_MET, "Driver Gene Dynamics with MET path", cmap="coolwarm", path=outputdir)

md_indices = HVG.get_indexer(md_gene)
driver_genes_md = HVG[md_indices]
MD_matrix = dynamic_driver_index_matrix_MET[:, md_indices].T
MD_norm = Detect_driver.row_normalize(MD_matrix)
Detect_driver.plot_phase_heatmap(MD_norm, 'purple', 2, driver_genes_md, "Driver Strength Dynamics of Brown Module", cluster_rows=False, cmap="coolwarm", path=outputdir)


from tqdm import tqdm
name1 = 'Beta'
with torch.no_grad():
    z, _ = best_model(All_data.float())
    z_min = z.min(dim=0, keepdim=True)[0]
    z_max = z.max(dim=0, keepdim=True)[0]
    z = scaled_output(z)
    data_t = z[target]
    data_t = pd.DataFrame(data_t.cpu().detach().numpy())
    data_s = z[source]
    data_s = pd.DataFrame(data_s.cpu().detach().numpy())
    C = cdist(data_s.values, data_t.values, metric='euclidean')

    if method == 'neighbor':
        mu = Get_Probability_Measures.Neighbor_Measures(data_s, 10, epsilon=1e-5)
        mu = mu.to_numpy()
        nu = Get_Probability_Measures.Neighbor_Measures(data_t, 10, epsilon=1e-5)
        nu = nu.to_numpy()
    else:
        mu = Get_Probability_Measures.kde_gene_expression(data_s)
        nu = Get_Probability_Measures.kde_gene_expression(data_t)
    P = ot.sinkhorn(mu, nu, C, reg=reg)
    P_normalized = P / P.sum(axis=1, keepdims=True)
    data_trans = np.dot(P_normalized, data_t.values)
    data_trans = torch.from_numpy(data_trans).to(device).to(torch.float32)
    data_s = torch.from_numpy(data_s.values).to(device)

    data_trans = data_trans * (z_max - z_min + 1e-8) + z_min
    data_s = data_s * (z_max - z_min + 1e-8) + z_min

    t_values = torch.linspace(0.0, 1.0, num_inter+1)
    mean_data1 = []
    for i, t in enumerate(t_values):
        with torch.no_grad():
            data_inter1 = (1 - t) * data_s + t * data_trans

            data_inter_org1 = inn(data_inter1.to(torch.float).to(device), rev=True)[0]
            mean_inter1 = data_inter_org1.mean(dim=0)

            mean_data1.append(mean_inter1)
    mean_data1 = torch.stack(mean_data1)
    t_np = t_values.cpu().numpy() * (num_inter/10)
    # Plot the scatter plot

    # Convert tensors to NumPy arrays for matplotlib
    filtered_data_np1 = mean_data1.detach().cpu().numpy()
    # filtered_data_np1[filtered_data_np1 < 0] = 0
    pd.DataFrame(filtered_data_np1).to_csv(f"{outputdir}/gene_dynamic/data.csv")
    # filtered_data_np2[filtered_data_np2 < 0] = 0
    matplotlib.use('Agg')
    # Plot a scatter plot for each feature
    for i in tqdm(range(filtered_data_np1.shape[1])):
        fig = plt.figure(figsize=(6, 4))
        # Plot filtered_data_np1 with label "MET"
        plt.scatter(t_np, filtered_data_np1[:, i], alpha=0.6, label=name1, marker='o', s=20, color='purple')

        gene_name = HVG[i]
        sanitized_gene_name = gene_name.replace('/', '_').replace('\\', '_')
        # Plot filtered_data_np2 with label "EMT"

        plt.title(sanitized_gene_name)  # Set the plot title to the feature name
        plt.xlabel("t")
        plt.ylabel(f"{sanitized_gene_name} expression")
        plt.grid(True)
        plt.legend()  # Show the legend
        # Save the image using the feature name as the filename
        plt.savefig(f"{outputdir}/gene_dynamic/{sanitized_gene_name}.png", format='png')
        plt.close(fig)  # Close the current figure to free memory



import Detect_driver
source = cluster_labels == 'Normal'
target = cluster_labels == 'T2D'
driver_index = Detect_driver.Detect_driver(All_data.float(), All_t*10, inn, np.ones(112), source=source, target=target, method='kde', reg=1e-1, name=1)
dynamic_driver_index_matrix = torch.stack(driver_index).cpu().numpy()
averaged_driver_index = torch.mean(torch.stack(driver_index), dim=0)

sorted_results = torch.sort(averaged_driver_index, descending=True)
sorted_driver_scores = sorted_results[0] # Sorted driver_index values
original_indices = sorted_results[1]   # Original indices corresponding to the sorted values

gene_names = pd.read_csv('your/path/to/data/exp.csv', index_col=0).T.columns
driver_genes_beta = gene_names[original_indices[0:100].cpu().detach().numpy()]
genes = pd.DataFrame(driver_genes_beta)
genes.to_csv('your/path/to/output/driver_genes_100.csv', index=False)




























