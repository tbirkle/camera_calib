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
    #if axis==0:
    #    other_axis = 1
    #elif axis==1:
    #    other_axis = 0

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



import skimage

def read_dir(path):
    flat = os.listdir(path)
    images = []
    for filename in flat:
        if (filename.split(".")[1] == "png"):
            img = scipy.misc.imread(os.path.join(path, filename), mode="I")
            #img = scipy.misc.toimage(img, low=0, high=4095, mode="I")
            #img = img.astype(np.uint16)
            images.append(img)
    images = np.asarray(images)

    return images



flat_images = read_dir("mv")
#dark_images = read_dir("dark")

print(len(flat_images))

dark1 = scipy.misc.imread(os.path.join("mv", "mv01-000a.png"), mode="I")
dark2 = scipy.misc.imread(os.path.join("mv", "mv01-000b.png"), mode="I")
dark_images = [dark1, dark2] * 51


print(np.mean(flat_images, axis=(1,2)))

mean_flat = averageEvery2(np.mean(flat_images, axis=(1,2)))
mean_dark = averageEvery2(np.mean(dark_images, axis=(1,2)))

print(mean_flat)
print(mean_dark)

var_flat = temporal_variance(flat_images)
var_dark = temporal_variance(dark_images)

exposure_times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15])

photons = number_photons(exposure_times)
photons = []
for i in range(51):
    photons.append(i * 1659.66)

photons = np.asarray(photons)

"""
max_idx = np.argmax(var_flat)
print("Saturation", mean_flat[max_idx], max_idx)

sat1 = mean_flat[max_idx] * 0.7

mean_reg_flat = mean_flat[mean_flat < sat1]
mean_reg_dark = mean_dark[:len(mean_reg_flat)]
reg_x = mean_reg_flat - mean_reg_dark
reg_y = var_flat[:len(mean_reg_flat)] - var_dark[:len(mean_reg_flat)]
regression = stats.linregress(reg_x, reg_y)
slope = regression[0]
"""

mean_without_noise = mean_flat - mean_dark
var_without_noise = var_flat - var_dark

saturation_idx = np.argmax(var_without_noise)
print("Saturation", mean_without_noise[saturation_idx], saturation_idx)

saturation_start = mean_without_noise[saturation_idx] * 0.7

regression_x = mean_without_noise[mean_without_noise < saturation_start]
regression_y = var_without_noise[:len(regression_x)]

print(regression_x)
print(regression_y)

regression = stats.linregress(regression_x, regression_y)

K = regression[0]
print("System gain K:", K)

plt.figure(1)
plt.title("Photon-Transfer-Curve")
plt.plot([0,saturation_start],[regression[1],regression[1]+regression[0]*saturation_start])
plt.plot(mean_without_noise, var_without_noise)
plt.plot([mean_without_noise[saturation_idx], mean_without_noise[saturation_idx]],[var_without_noise[saturation_idx],0],"--")

"""Full-well capacity"""
max_var = max(var_flat - var_dark)  # TODO ist das richtig?
print("Full-Well:", max_var)




"""Sensitivity"""
reg_y = mean_without_noise[mean_without_noise < saturation_start]
reg_x = photons[:len(reg_y)]
regression = stats.linregress(reg_x, reg_y)
slope = regression[0]

R = slope
print("Responsitivity R: ", R)


plt.figure(2)
plt.title("Sensitivity")
plt.plot(photons, mean_without_noise)
plt.plot([0,reg_x[len(reg_x)-1]],[regression[1], regression[1]+regression[0]*reg_x[len(reg_x)-1]], "r--")


"""Quantum Efficiency"""
QE = R / K
print("Quantum Efficiency", QE)

"""Dark Noise"""

var_dark_temp = var_dark[0]
"""
if var_dark_temp < 0.24: # temporal noise is dominated by the quantization noise P.17 EMVA
    var_dark_temp = 0.49
    dark_noise = 0.4/K
else:
    dark_noise = np.sqrt(var_dark_temp-QUANTISATION_ERROR) / K
"""

if var_dark_temp < 0.24: # temporal noise is dominated by the quantization noise P.17 EMVA
    var_dark_temp = 0.49

var_dark_noise = (var_dark_temp - QUANTISATION_ERROR) / K**2

print("Variance Dark Noise: ", var_dark_noise, "e")
print("Std Dark Noise: ", np.sqrt(var_dark_noise), "e")

"""AUFGABE 1 ENDE"""


"""Signal to Noise ratio"""
SNR_each_pic = (mean_without_noise)/np.sqrt(var_flat)  #SNR für jede Belichtungszeit

SNR_theor = (QE*photons)/np.sqrt(var_dark+(1/12/K**2)+QE*photons) #SNR aus den zuvor ermittelten Werten

SNR_ideal = np.sqrt(photons) #SNR von idealem Sensor

sat_photons = photons[len(mean_without_noise[mean_without_noise < saturation_start])-1]

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
plt.plot([sat_photons, sat_photons],[0, SNR_ideal.max()+100], "--")
plt.plot([photons_min, photons_min],[0, SNR_ideal.max()+100], "--")
plt.plot()

plt.legend(handles=[ideal, theo, real])


"""Dynamikbereich"""

DR = 20*np.log10((sat_photons/photons_min))
print("Dynamikbereich: ", DR)

#plt.show()



"""Ende Aufgabe 2"""

#flat50 = read_dir("50flat")
flat50 = read_dir("prnu")
mean_flat50_image = np.mean(flat50, axis=0)
mean_flat50 = mean_flat50_image.mean()
size = mean_flat50_image.shape[0] * mean_flat50_image.shape[1]

variance_flat50 = (1/size) * (np.sum((mean_flat50_image - mean_flat50)**2))




#dark50 = read_dir("50dark")
dark50 = read_dir("dsnu")
mean_dark50_image = np.mean(dark50, axis=0)
mean_dark50 = mean_dark50_image.mean()
size = mean_dark50_image.shape[0] * mean_dark50_image.shape[1]
variance_dark50 = (1/size) * (np.sum((mean_dark50_image - mean_dark50)**2))

temporal_variance_flat50_image = (1/(len(flat50)-1)) * np.sum((flat50 - mean_flat50_image)**2,axis=0)
temporal_variance_flat50_stack = np.mean(temporal_variance_flat50_image)

std_flat50_stack = np.sqrt(temporal_variance_flat50_stack)

temporal_variance_dark50_image = (1/(len(dark50)-1)) * np.sum((dark50 - mean_dark50_image)**2,axis=0)
temporal_variance_dark50_stack = np.mean(temporal_variance_dark50_image)
std_dark50_stack = np.sqrt(temporal_variance_dark50_stack)

DSNU = variance_dark50 / K
PRNU = np.sqrt(variance_flat50 - variance_dark50) / (mean_flat50 - mean_dark50)

print("DSNU: ", DSNU, "e")
print("PRNU: ", PRNU, "%")







# mean_flat50_image =  mean_flat50_image - mean_flat50
mean_flat50_image_diff = mean_flat50_image - mean_flat50
mean_dark50_image_diff = mean_dark50_image - mean_dark50

"""Spektrogramm PRNU Horizontal"""
flat_fft_hor = (1/np.sqrt(mean_flat50_image_diff.size)) * np.fft.fft(mean_flat50_image_diff, axis=0)
p_flat_hor = power_spectrum(flat_fft_hor, axis=0)

plt.figure(4)
plt.title("Spektrogramm PRNU Horizontal")
plt.yscale("log")
plt.plot(p_flat_hor[:len(p_flat_hor)//2])
plt.plot([0, len(p_flat_hor)//2],[std_flat50_stack,std_flat50_stack], "g--")
plt.plot([0, len(p_flat_hor)//2],[PRNU,PRNU], "r--")

#plt.plot(range(len(p)//2), p[:len(p)//2])
"""Spektrogramm PRNU Vertikal"""

flat_fft_vert = (1/np.sqrt(mean_flat50_image_diff.size)) * np.fft.fft(mean_flat50_image_diff, axis=1)
p_flat_vert = power_spectrum(flat_fft_vert, axis=1)
plt.figure(5)
plt.title("Spektrogramm PRNU Vertikal")
plt.yscale("log")
plt.plot(p_flat_vert[:len(p_flat_vert)//2])
plt.plot([0, len(p_flat_vert)//2],[std_flat50_stack,std_flat50_stack], "g--")
plt.plot([0, len(p_flat_vert)//2],[PRNU,PRNU], "r--")

"""Spektrogramm DSNU Horizontal"""
dark_fft_hor = (1/np.sqrt(mean_dark50_image_diff.size)) * np.fft.fft(mean_dark50_image_diff, axis=0)
p_dark_hor = power_spectrum(dark_fft_hor, axis=0)
plt.figure(6)
plt.title("Spektrogramm DSNU Horizontal")
plt.yscale("log")
plt.plot(p_dark_hor[:len(p_dark_hor)//2])
plt.plot([0, len(p_dark_hor)//2],[std_dark50_stack,std_dark50_stack], "g--")
plt.plot([0, len(p_dark_hor)//2],[DSNU,DSNU], "r--")

"""Spektrogramm DSNU Vertikal"""
dark_fft_vert = (1/np.sqrt(mean_dark50_image_diff.size)) * np.fft.fft(mean_dark50_image_diff, axis=1)
p_dark_vert = power_spectrum(dark_fft_vert, axis=1)
plt.figure(7)
plt.title("Spektrogramm DSNU Vertikal")
plt.yscale("log")
plt.plot(p_dark_vert[:len(p_dark_vert)//2])
plt.plot([0, len(p_dark_vert)//2],[std_dark50_stack,std_dark50_stack], "g--")
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
