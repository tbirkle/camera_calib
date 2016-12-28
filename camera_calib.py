import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.misc
import os


SENSOR_SIZE = 0.00045
E = 123
WAVELENGTH = 0.000420
CONSTANT = 5.034*10.24
QUANTISATION_ERROR = 1/12




def number_photons(exposure_times):
    return CONSTANT * SENSOR_SIZE * E * exposure_times * WAVELENGTH

def averageEvery2(x):
    mean_flat = np.vstack((x[0::2], x[1::2]))
    return np.mean(mean_flat, axis=0)

def temporal_variance(x):
    y_a = x[0::2]
    y_b = x[1::2]


    size = 2* y_a[0].shape[0] * y_a[0].shape[1]

    variance = (1/size) * np.sum(np.subtract(y_a,y_b)**2, axis=(1,2))
    return variance

from scipy.ndimage import gaussian_filter
from scipy.misc import imsave
"""
def test(path):
    flat = os.listdir(path)
    for filename in flat:
        img = plt.imread(os.path.join(path, filename))
        print(img.shape)
        img = gaussian_filter(img, sigma=3)
        print(img.shape)
        imsave(os.path.join(path, filename.split(".")[0]+"_2.png"), img)

"""

def read_dir(path):
    flat = os.listdir(path)
    images = []
    std = np.array([])
    mean = np.array([])
    for filename in flat:
        img = scipy.misc.imread(os.path.join(path, filename))
        print(img.shape)
        images.append(img)
        std = np.append(std, img.std())
        mean = np.append(mean, img.mean())


    images = np.asarray(images)
    #var = std ** 2


    # Averaging every 2 values

    mean = averageEvery2(mean)
    var = temporal_variance(images)
    #std = averageEvery2(std)
    #var = averageEvery2(var)

    return images, std, mean, var



flat_images, std_flat, mean_flat, var_flat = read_dir("flat")
dark_images, std_dark, mean_dark, var_dark = read_dir("dark")


print(var_flat)
print(var_dark)

exp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15])

phot = number_photons(exp)



#Ill_intens = [Bel-staerke1, Bel-saerke2,...]
#Ill_time = [Bel-zeit1, Bel-zeit2,...]
#phot_num = 5.034 * 10^24 # Sernsorflaeche_muss_noch_nachgeschaut_werden #Ill_intens[:len(Ill_intens)]  Ill_time[:len(Ill_intens)]

max_idx = np.argmax(var_flat)
print("mean_flat", mean_flat)
print("Saturation", mean_flat[max_idx], max_idx)

sat1 = mean_flat[max_idx] * 0.7

mean_reg_flat = mean_flat[mean_flat < sat1]
mean_reg_dark = mean_dark[:len(mean_reg_flat)]

reg_x = mean_reg_flat - mean_reg_dark
reg_y = var_flat[:len(mean_reg_flat)] - var_dark[:len(mean_reg_flat)]

regression = stats.linregress(reg_x, reg_y)
slope = regression[0]
K = slope

print("System gain K:", K)
plt.figure(1)
plt.plot([0,sat1],[regression[1],regression[1]+regression[0]*sat1])


print(var_flat)
print(max_idx)
max_var = max(var_flat - var_dark)
print("Full-Well:", max_var)



plt.plot(mean_flat - mean_dark, var_flat - var_dark)


"""Linearity"""
reg_y = mean_reg_flat - mean_reg_dark
reg_x = phot[:len(reg_y)]
regression = stats.linregress(reg_x, reg_y)
slope = regression[0]


plt.figure(2)
plt.plot(phot, mean_flat-mean_dark)

plt.plot([0,reg_x[len(reg_x)-1]],[regression[1], regression[1]+regression[0]*reg_x[len(reg_x)-1]], "-")
R = slope

print("Responsitivity R: ", R)
plt.show()

"""Quantum Efficiency"""
QE = R / K
print("Quantum Efficiency", QE)

"""Dark Noise"""
dark_noise = (var_dark-QUANTISATION_ERROR) / K**2


"""AUFGABE 1 ENDE"""


#plt.imshow(flat_images[10], cmap='Greys')
#plt.show()

