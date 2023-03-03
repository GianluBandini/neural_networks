import numpy as np
import matplotlib.pyplot as plt

# Generate some random data
data = np.random.rand(5, 5)

# Plot the heatmap
fig, ax = plt.subplots()
im = ax.imshow(data)

# Show the values in each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = ax.text(j, i, round(data[i, j], 2), ha="center", va="center", color="w")

# Set the x-axis and y-axis ticks
ax.set_xticks(np.arange(data.shape[1]))
ax.set_yticks(np.arange(data.shape[0]))

# Set the x-axis and y-axis tick labels
ax.set_xticklabels(np.arange(1, data.shape[1]+1))
ax.set_yticklabels(np.arange(1, data.shape[0]+1))

# Set the plot title and color bar
ax.set_title("Heatmap with Values")
cbar = ax.figure.colorbar(im, ax=ax)

plt.show()