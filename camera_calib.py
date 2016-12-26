import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import os


def averageEvery2(x):
    mean_flat = np.vstack((x[0::2], x[1::2]))
    return np.mean(mean_flat, axis=0)

def read_dir(path):
    flat = os.listdir(path)
    images = []
    std = np.array([])
    mean = np.array([])
    for filename in flat:
        img = plt.imread(os.path.join(path, filename))
        images.append(img)
        std = np.append(std, img.std())
        mean = np.append(mean, img.mean())

    var = std ** 2


    # Averaging every 2 values
    mean = averageEvery2(mean)
    std = averageEvery2(std)
    var = averageEvery2(var)

    return images, std, mean, var



flat_images, std_flat, mean_flat, var_flat = read_dir("flat")
dark_images, std_dark, mean_dark, var_dark = read_dir("dark")


max_idx = np.argmax(var_flat)
print("Saturation", mean_flat[max_idx])

sat1 = mean_flat[max_idx] * 0.7
mean_reg_flat = mean_flat[mean_flat < sat1]

mean_reg_dark = mean_dark[:len(mean_reg_flat)]

reg_x = mean_reg_flat - mean_reg_dark
reg_y = var_flat[:len(mean_reg_flat)] - var_dark[:len(mean_reg_flat)]
regression = stats.linregress(reg_x, reg_y)
slope = regression[0]
print("System gain K:", slope)


plt.plot([0,sat1],[regression[1],regression[1]+regression[0]*sat1])


print(var_flat)
print(max_idx)
max_var = max(var_flat - var_dark)
print("Full-Well:", max_var)



plt.plot(mean_flat - mean_dark, var_flat - var_dark)
plt.show()







print(flat_images[9].min())
print(flat_images[9].max())
print(flat_images[9].mean())
print(flat_images[9].std())




#plt.imshow(flat_images[10], cmap='Greys')
#plt.show()

