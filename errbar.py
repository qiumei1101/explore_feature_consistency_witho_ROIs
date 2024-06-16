# Below is a Python script using Matplotlib to create a bar chart that visualizes mean values with error bars representing the standard deviations. This script assumes you have two sets of 8 mean values and their corresponding standard deviations—one set where both vehicles are inside the ROI and another where one vehicle is inside the ROI and the other is outside.

# Here's how you can set up and plot this data:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = 10
# resnet50 data
# mean_values_inside = np.array([0.7427, 0.7074, 0.7427, 0.8287, 0.7842, 0.7632, 0.8008, 0.7351])  # Mean values with both vehicles inside the ROI
# std_dev_inside = np.array([0.0949, 0.1046, 0.0858, 0.0547, 0.0943, 0.0846, 0.0987, 0.1223])      # Standard deviations for the above means

# mean_values_cross = np.array([0.7282, 0.6964, 0.7195, 0.7868, 0.7181, 0.7344, 0.7677, 0.7438])  # Mean values with one vehicle inside and one outside the ROI
# std_dev_cross = np.array([0.0901, 0.0951, 0.0953, 0.0659, 0.0810, 0.0849, 0.1010, 0.1136])      # Standard deviations for the above means

#vit data
mean_values_inside = np.array([0.4070, 0.3499, 0.5119, 0.5631, 0.2635, 0.3001, 0.3499, 0.3349])  # Mean values with both vehicles inside the ROI
std_dev_inside = np.array([0.1611, 0.2033, 0.1671, 0.1601, 0.1491, 0.1640, 0.1689, 0.1604])      # Standard deviations for the above means

mean_values_cross = np.array([0.3557, 0.3042, 0.4211, 0.3972, 0.1826, 0.2371, 0.3006, 0.2600])  # Mean values with one vehicle inside and one outside the ROI
std_dev_cross = np.array([0.1589, 0.1786, 0.1824, 0.1830, 0.1197, 0.1257, 0.1496, 0.1397])      # Standard deviations for the above means

# X positions of bars
x = np.arange(len(mean_values_inside))

# Width of a bar
width = 0.35  

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, mean_values_inside, width, label='Both Inside ROI', yerr=std_dev_inside, capsize=5)
bars2 = ax.bar(x + width/2, mean_values_cross, width, label='One Inside, One Outside ROI', yerr=std_dev_cross, capsize=5)

# Adding labels and title
ax.set_xlabel('Video Conditions')
ax.set_ylabel('Mean Cosine Similarity with ViTB-16')
ax.set_title('Mean Cosine Similarity Values and \n Standard Deviations by ROI Configuration',fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Congestion', 'Congestion2', 'Sunny', 'Sunny2', 'Night', 'Night2', 'Rainy', 'Rainy2'],rotation=15)
ax.legend()

# Adding some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xticks(x)
# ax.set_xticklabels(['Congestion', 'Congestion2', 'Sunny', 'Sunny2', 'Night', 'Night2', 'Rainy', 'Rainy2'])

# plt.show()
plt.savefig("cs_sm_vitb-16.png")

