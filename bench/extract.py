import numpy as np

# Read the data from the file
with open("bench/benchfileOMP", "r") as f:
    values = [float(x) for x in f.read().split()]

# Organize the data into pairs: time, index
data = np.array(values).reshape(-1, 2)

times = data[:, 0]
indices = data[:, 1]

# Calculate the mean time
sum_times = np.sum(times)
n_times = len(times)
mean_time = sum_times / n_times
time_variance = np.var(times)

# Calculate the mean index
sum_indices = np.sum(indices)
n_indices = len(indices)
mean_index = sum_indices / n_indices
mlups_variance = np.var(indices)


print("MEAN TIME")
print(f"Sum = {sum_times:.5f}")
print(f"N = {n_times}")
print(f"Mean = {sum_times:.5f} / {n_times} = {mean_time:.5f}")

print(f"VARIANCE = {time_variance}")


print("\nMEAN MLUPS")
print(f"Sum = {sum_indices:.5f}")
print(f"N = {n_indices}")
print(f"Mean = {sum_indices:.5f} / {n_indices} = {mean_index:.5f}")
print(f"VARIANCE: {mlups_variance}")
