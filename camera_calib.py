import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.misc
import os


PIXEL_SIZE = (4.5 * 10**(-6))**2
E = 123
WAVELENGTH = 0.000420
CONSTANT = 5.034* 10**24
QUANTISATION_ERROR = 1/12




def number_photons(exposure_times):
    return CONSTANT * PIXEL_SIZE * E * exposure_times * WAVELENGTH

def averageEvery2(x):
    mean_flat = np.vstack((x[0::2], x[1::2]))
    return np.mean(mean_flat, axis=0)

def temporal_variance(x):
    y_a = x[0::2]
    y_b = x[1::2]

    size = 2* y_a[0].shape[0] * y_a[0].shape[1]

    variance = (1/size) * np.sum(np.subtract(y_a,y_b)**2, axis=(1,2))
    return variance

def power_spectrum(fft, axis=0):
    return np.sqrt((1 / fft.shape[~axis]) * np.sum((fft * np.conj(fft)), axis=~axis))

from scipy.ndimage import gaussian_filter
from scipy.misc import imsave
"""
def test(path):
    flat = os.listdir(path)
    for filename in flat:
        if(filename.split(".")[1] == "png"):
            img = scipy.misc.imread(os.path.join(path, filename))
            print(img.shape)
            #img = gaussian_filter(img, sigma=3)
            print(img.shape)
            test1 = np.random.normal(-1, 1, img.shape)
            test2 = np.random.normal(1, 2, img.shape)

            imsave(os.path.join(path, filename.split(".")[0] + ".png"), img + test1)
            imsave(os.path.join(path, filename.split(".")[0]+"_2.png"), img + test2)
"""




def read_dir(path):
    flat = os.listdir(path)
    images = []
    for filename in flat:
        if (filename.split(".")[1] == "png"):
            img = scipy.misc.imread(os.path.join(path, filename))
            images.append(img)

    images = np.asarray(images)

    return images



flat_images = read_dir("flat")
dark_images = read_dir("dark")

mean_flat = averageEvery2(np.mean(flat_images, axis=(1,2)))
mean_dark = averageEvery2(np.mean(dark_images, axis=(1,2)))

var_flat = temporal_variance(flat_images)
var_dark = temporal_variance(dark_images)

exposure_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15])

photons = number_photons(exposure_times)


max_idx = np.argmax(var_flat)
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
plt.title("Photon-Transfer-Curve")
plt.plot([0,sat1],[regression[1],regression[1]+regression[0]*sat1])
plt.plot(mean_flat - mean_dark, var_flat - var_dark)


"""Full-well capacity"""
max_var = max(var_flat - var_dark)
print("Full-Well:", max_var)






"""Sensitivity"""
reg_y = mean_reg_flat - mean_reg_dark
reg_x = photons[:len(reg_y)]
regression = stats.linregress(reg_x, reg_y)
slope = regression[0]

R = slope
print("Responsitivity R: ", R)


plt.figure(2)
plt.title("Sensitivity")
plt.plot(photons, mean_flat-mean_dark)
plt.plot([0,reg_x[len(reg_x)-1]],[regression[1], regression[1]+regression[0]*reg_x[len(reg_x)-1]], "-")


"""Quantum Efficiency"""
QE = R / K
print("Quantum Efficiency", QE)

"""Dark Noise"""
dark_noise = (var_dark[0]-QUANTISATION_ERROR) / K**2
print("Dark Noise: ", dark_noise)

"""AUFGABE 1 ENDE"""


"""Signal to Noise ratio"""
SNR_each_pic = (mean_flat-mean_dark)/np.sqrt(var_flat)  #SNR für jede Belichtungszeit

SNR_theor = (QE*photons)/np.sqrt(var_dark+(1/12/K**2)+QE*photons) #SNR aus den zuvor ermittelten Werten

SNR_ideal = np.sqrt(photons) #SNR von idealem Sensor

sat_photons = photons[len(mean_reg_flat)-1]

photons_min = (1/QE) *((np.sqrt(var_dark[0])/K)+0.5) #Wie berechnet man var_dark ? Welche Bilder (Belichtungszeiten)?


plt.figure(3)
plt.title("SNR")
#ax = fig3.axes()
plt.yscale("log")
plt.xscale("log")

ideal, = plt.plot(photons, SNR_ideal,"r",label="ideale SNR")
theo, = plt.plot(photons, SNR_theor,"g", label="theoretische SNR")
real, = plt.plot(photons, SNR_each_pic,"bx", label="reale SNR")
#plt.plot[[sat_photons,0],[sat_photons,10**10]]
plt.plot([sat_photons, sat_photons],[0, 100000000], "--")
plt.plot([photons_min, photons_min],[0, 100000000], "--")
plt.plot()

plt.legend(handles=[ideal, theo, real])


"""Dynamikbereich"""

DR = 20*np.log10((sat_photons/photons_min))
print("Dynamikbereich: ", DR)

#plt.show()



"""Ende Aufgabe 2"""

flat50 = read_dir("50flat")
mean_flat50_image = np.mean(flat50, axis=0)
mean_flat50 = mean_flat50_image.mean()
size = mean_flat50_image.shape[0] * mean_flat50_image.shape[1]

variance_flat50 = (1/size) * (np.sum((mean_flat50_image - mean_flat50)**2))




dark50 = read_dir("50dark")
mean_dark50_image = np.mean(dark50, axis=0)
mean_dark50 = mean_dark50_image.mean()
size = mean_dark50_image.shape[0] * mean_dark50_image.shape[1]
variance_dark50 = (1/size) * (np.sum((mean_dark50_image - mean_dark50)**2))

temporal_variance_flat50_image = (1/(len(flat50)-1)) * np.sum(flat50 - mean_flat50_image,axis=0)
temporal_variance_flat50_stack = np.mean(temporal_variance_flat50_image)

temporal_variance_dark50_image = (1/(len(dark50)-1)) * np.sum(dark50 - mean_dark50_image,axis=0)
temporal_variance_dark50_stack = np.mean(temporal_variance_dark50_image)


DSNU = variance_dark50 / K
PRNU = np.sqrt(variance_flat50 - variance_dark50) / (mean_flat50 - mean_dark50)

print("DSNU: ", DSNU, "e")
print("PRNU: ", PRNU, "%")





# mean_flat50_image =  mean_flat50_image - mean_flat50
mean_flat50_image_diff = mean_flat50_image - mean_flat50
mean_dark50_image_diff = mean_dark50_image - mean_dark50

"""Spektrogramm PRNU Horizontal"""
flat_fft_hor = np.fft.fft(mean_flat50_image_diff, axis=0)
p_flat_hor = power_spectrum(flat_fft_hor, axis=0)
plt.title("Spektrogramm PRNU Horizontal")
plt.figure(4)
plt.yscale("log")
plt.plot(p_flat_hor[:len(p_flat_hor)//2])
plt.plot([0, len(p_flat_hor)//2],[temporal_variance_flat50_stack,temporal_variance_flat50_stack], "g--")
plt.plot([0, len(p_flat_hor)//2],[PRNU,PRNU], "r--")

#plt.plot(range(len(p)//2), p[:len(p)//2])
"""Spektrogramm PRNU Vertikal"""
flat_fft_vert = np.fft.fft(mean_flat50_image_diff, axis=1)
p_flat_vert = power_spectrum(flat_fft_vert, axis=1)
plt.figure(5)
plt.title("Spektrogramm PRNU Vertikal")
plt.yscale("log")
plt.plot(p_flat_vert[:len(p_flat_vert)//2])
plt.plot([0, len(p_flat_vert)//2],[temporal_variance_flat50_stack,temporal_variance_flat50_stack], "g--")
plt.plot([0, len(p_flat_vert)//2],[PRNU,PRNU], "r--")

"""Spektrogramm DSNU Horizontal"""
dark_fft_hor = np.fft.fft(mean_dark50_image_diff, axis=0)
p_dark_hor = power_spectrum(dark_fft_hor, axis=0)
plt.figure(6)
plt.title("Spektrogramm DSNU Horizontal")
plt.yscale("log")
plt.plot(p_dark_hor[:len(p_dark_hor)//2])
plt.plot([0, len(p_dark_hor)//2],[temporal_variance_dark50_stack,temporal_variance_dark50_stack], "g--")
plt.plot([0, len(p_dark_hor)//2],[DSNU,DSNU], "r--")

"""Spektrogramm DSNU Vertikal"""
dark_fft_vert = np.fft.fft(mean_dark50_image_diff, axis=1)
p_dark_vert = power_spectrum(dark_fft_vert, axis=1)
plt.figure(7)
plt.title("Spektrogramm DSNU Vertikal")
plt.yscale("log")
plt.plot(p_dark_vert[:len(p_dark_vert)//2])
plt.plot([0, len(p_dark_vert)//2],[temporal_variance_dark50_stack,temporal_variance_dark50_stack], "g--")
plt.plot([0, len(p_dark_vert)//2],[DSNU,DSNU], "r--")

"""ENDE AUFGABE3"""

"""AUFGABE 4"""
lowpass = scipy.ndimage.uniform_filter(mean_flat50_image - mean_dark50_image, 5, mode="reflect")
print(lowpass.shape)

highpass = mean_flat50_image - mean_dark50_image - lowpass

plt.figure(8)
plt.title("histogram PRNU")
plt.yscale("log")
plt.hist(highpass.flatten())

plt.figure(9)
plt.title("histogram DSNU")
plt.yscale("log")
plt.hist(mean_dark50_image.flatten())


plt.show()
