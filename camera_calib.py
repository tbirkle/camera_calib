import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.misc
import scipy.ndimage
import os
import cv2

from matplotlib.patches import Circle
import skimage.io


#50% sat = 8.58ms
"""
img = scipy.misc.imread("flat/"+"flat_24.png")

img = np.asarray((img / 2 ** 16) * 2 ** 12, dtype=np.uint16)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
img = img_rgb[:,:,1]

plt.figure(1)
plt.imshow(img_rgb)

plt.figure(2)
scipy.misc.imsave("testrot.png",img_rgb[:,:,0])


plt.figure(3)
scipy.misc.imsave("testgr.png", img_rgb[:,:,1])

plt.figure(4)
scipy.misc.imsave("testbl.png", img_rgb[:,:,2])

exit()
plt.show()
"""
PIXEL_SIZE = (4.5 * 10**(-6))**2
E = 0.1217 #13.97 # 46.85nA / 3.352nA/lx = 13.97lx
WAVELENGTH = 0.000000525
CONSTANT = 5.034* 10**24
QUANTISATION_ERROR = 1/12

BIT = 12

#print(np.linspace(0.02, 20, 15))

#exit()

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

def flat_fielding(mean_flat50_image, mean_dark50_image, test_image):
    return np.mean(mean_flat50_image) * ((test_image - mean_dark50_image)/(mean_flat50_image-mean_dark50_image))


def read_dir(path):
    flat = sorted(os.listdir(path))
    images = []
    for filename in flat:
        if (filename.split(".")[1] == "png"):
            #print(filename)
            img = scipy.misc.imread(os.path.join(path, filename))
            #print(img.shape, img.dtype, img.max(), img.min(), img.mean())

            img = np.asarray((img / 2**16) * 2**BIT, dtype=np.uint16)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
            img = img_rgb[:,:,1]

            #print(img.shape, img.dtype, img.max(), img.min(), img.mean())

            #img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
            #scipy.misc.imsave("test1234.png", img)#, cmap="Greys")#[:,:,1], cmap="Greys")
           # print(img[:,:,2])
            #print(img.shape, img.dtype, img.max(), img.min(), img.mean())
            #plt.show()
            #exit()
            images.append(img)
    images = np.asarray(images)

    return images



flat_images = read_dir("flat")
dark_images = read_dir("dark")

print(len(flat_images))

"""
dark1 = scipy.misc.imread(os.path.join("mv", "mv01-000a.png"), mode="I")
dark1 = np.asarray((dark1 / 255) * 2 ** BIT - 1, dtype=np.uint16)

dark2 = scipy.misc.imread(os.path.join("mv", "mv01-000b.png"), mode="I")
dark2 = np.asarray((dark2 / 255) * 2 ** BIT - 1, dtype=np.uint16)

dark_images = [dark1, dark2] * 51
"""
print(flat_images.shape)
print(np.mean(flat_images, axis=(1,2)))

test = np.mean(flat_images, axis=(1,2))
print(test.shape)

mean_flat = averageEvery2(np.mean(flat_images, axis=(1,2)))
mean_dark = averageEvery2(np.mean(dark_images, axis=(1,2)))


print(mean_flat.shape)


print(mean_flat)
print(mean_dark)

var_flat = temporal_variance(flat_images)
var_dark = temporal_variance(dark_images)

exposure_times = np.linspace(0.02, 20, 15) / 1000#np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15])
print("exposure times",exposure_times)
photons = number_photons(exposure_times)

print("photons", photons)


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
K_offset = regression[1]
K_error = regression[4]
print("System gain K:", K)

"""Full-well capacity"""
max_var = max(var_flat - var_dark)  # TODO ist das richtig?
print("Full-Well:", max_var)




"""Sensitivity"""
reg_y = mean_without_noise[mean_without_noise < saturation_start]
reg_x = photons[:len(reg_y)]
regression = stats.linregress(reg_x, reg_y)

R = regression[0]
R_offset = regression[1]
R_error = regression[4]
print("Responsitivity R: ", R)

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



plt.figure(1)
plt.title("Photon-Transfer-Curve")
plt.ylabel(r'variance gray value  $\sigma_{y} - \sigma_{y.dark}$  $(DN^2)$')
plt.xlabel(r'gray value - dark value  $\mu_{y} - \mu_{y.dark}$  $(DN)$')
plt.text(50, 0.1, r'$\sigma^2_{{y.dark}}={:.2f} DN^2, K={:.3f} \pm{:.3f}\%$'.format(var_dark_noise, K, K_error))
plt.plot([0,saturation_start],[K_offset,K_offset+K*saturation_start], "r--", label="fit")
plt.plot(mean_without_noise, var_without_noise, "k+", label="data")
plt.plot([mean_without_noise[saturation_idx], mean_without_noise[saturation_idx]],[var_without_noise[saturation_idx],0],"b--", label="sensitivity threshold" )
plt.legend(loc="upper left")

plt.figure(2)
plt.title("Sensitivity")
plt.xlabel(r'irradiation (photons/pixel)')
plt.ylabel(r'gray value - dark value  $\mu_{y} - \mu_{y.dark}$  $(DN)$')
plt.text(1, 100, r"$\mu_{{y.dark}}={:.2f} DN$".format(mean_dark[0]))
plt.plot(photons, mean_without_noise, "k+", label="data")
plt.plot([0,reg_x[len(reg_x)-1]],[R_offset, R_offset+R*reg_x[len(reg_x)-1]], "r--", label="fit")
plt.legend(loc="upper left")






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

plt.legend(handles=[ideal, theo, real], loc='upper left')


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
flat_fft_hor = (1/np.sqrt(mean_flat50_image_diff.size)) * np.fft.fft(mean_flat50_image_diff, axis=1)
p_flat_hor = power_spectrum(flat_fft_hor, axis=1)

plt.figure(4)
plt.title("Spektrogramm PRNU Horizontal")
plt.ylabel("standard deviation(%)")
plt.xlabel("frequency (cycles/pixel)")
plt.yscale("log")
plt.plot(p_flat_hor[:len(p_flat_hor)//2])
plt.plot([0, len(p_flat_hor)//2],[std_flat50_stack,std_flat50_stack], "g--", label="temp.std {:.2f} DN".format(std_flat50_stack))
plt.plot([0, len(p_flat_hor)//2],[PRNU,PRNU], "r--", label="spat.std {:.2f} DN".format(PRNU))
plt.legend(loc="upper left")

#plt.plot(range(len(p)//2), p[:len(p)//2])
"""Spektrogramm PRNU Vertikal"""

flat_fft_vert = (1/np.sqrt(mean_flat50_image_diff.size)) * np.fft.fft(mean_flat50_image_diff, axis=0)
p_flat_vert = power_spectrum(flat_fft_vert, axis=0)
plt.figure(5)
plt.title("Spektrogramm PRNU Vertikal")
plt.ylabel("standard deviation(%)")
plt.xlabel("frequency (cycles/pixel)")
plt.yscale("log")
plt.plot(p_flat_vert[:len(p_flat_vert)//2])
plt.plot([0, len(p_flat_vert)//2],[std_flat50_stack,std_flat50_stack], "g--", label="temp.std {:.2f} DN".format(std_flat50_stack))
plt.plot([0, len(p_flat_vert)//2],[PRNU,PRNU], "r--", label="spat.std {:.2f} DN".format(PRNU))
plt.legend(loc="upper left")

"""Spektrogramm DSNU Horizontal"""
dark_fft_hor = (1/np.sqrt(mean_dark50_image_diff.size)) * np.fft.fft(mean_dark50_image_diff, axis=1)
p_dark_hor = power_spectrum(dark_fft_hor, axis=1)
plt.figure(6)
plt.title("Spektrogramm DSNU Horizontal")
plt.ylabel("standard deviation(DN)")
plt.xlabel("frequency (cycles/pixel)")
plt.yscale("log")
plt.plot(p_dark_hor[:len(p_dark_hor)//2])
plt.plot([0, len(p_dark_hor)//2],[std_dark50_stack,std_dark50_stack], "g--", label="temp.std {:.2f} DN".format(std_dark50_stack))
plt.plot([0, len(p_dark_hor)//2],[DSNU,DSNU], "r--", label="spat.std {:.2f} DN".format(DSNU))
plt.legend(loc="upper left")

"""Spektrogramm DSNU Vertikal"""
dark_fft_vert = (1/np.sqrt(mean_dark50_image_diff.size)) * np.fft.fft(mean_dark50_image_diff, axis=0)
p_dark_vert = power_spectrum(dark_fft_vert, axis=0)
plt.figure(7)
plt.title("Spektrogramm DSNU Vertikal")
plt.ylabel("standard deviation(DN)")
plt.xlabel("frequency (cycles/pixel)")
plt.yscale("log")
plt.plot(p_dark_vert[:len(p_dark_vert)//2])
plt.plot([0, len(p_dark_vert)//2],[std_dark50_stack,std_dark50_stack], "g--", label="temp.std {:.2f} DN".format(std_dark50_stack))
plt.plot([0, len(p_dark_vert)//2],[DSNU,DSNU], "r--", label="spat.std {:.2f} DN".format(DSNU))
plt.legend(loc="upper left")

"""ENDE AUFGABE3"""

"""AUFGABE 4"""
lowpass = scipy.ndimage.uniform_filter(mean_flat50_image - mean_dark50_image, 5, mode="reflect")


highpass = mean_flat50_image - mean_dark50_image - lowpass

#highpass_hist
"""
highpass_var = highpass.var()
total_var = highpass_var + temporal_variance_flat50_stack


mean_highpass = highpass.mean()

left = stats.norm.ppf(0.01, mean_highpass, highpass_var)
right = stats.norm.ppf(0.99, mean_highpass, highpass_var)

x = np.linspace(left, right, 100)
fit = stats.norm.pdf(x, mean_highpass, total_var)

mean_flat50_min = mean_flat50_image.min()
mean_flat50_max = mean_flat50_image.max()
L = len(flat50)
print("L: ", L)

interval = np.floor((L * (mean_flat50_max - mean_flat50_min))/256) + 1
print("Interval: ", interval)
Q = np.floor((L * (mean_flat50_max - mean_flat50_min)) / interval) + 1
print("Q: ", Q)

highpass_hist = np.histogram(highpass.flatten(), bins=Q, density=True, range=(highpass.min(), highpass.max()))
print("min", mean_flat50_min, mean_flat50_image.mean(), mean_flat50_image.max()) #(highpass_hist[0]* (1/L))
print(np.asarray([x for x in range(0,len(highpass_hist[0]))]))
center_bins = mean_flat50_min + ((interval - 1) / (2*L)) + (np.asarray([x for x in range(0,len(highpass_hist[0]))])* (interval/L))
print(center_bins.shape, center_bins)


print("size", mean_flat50_image.size )

print("asdf", (mean_flat50_image.size / np.sqrt(2*np.pi) * std_dark50_stack))
print("asdf", temporal_variance_dark50_stack)
print("asdfadvas", np.exp(-1 * (center_bins**2) / (2*temporal_variance_dark50_stack)))

print("E INNEN", -1 * ((center_bins**2) / (2*temporal_variance_dark50_stack)))

print(std_dark50_stack, temporal_variance_dark50_stack,  (interval/L))
pdf = (interval/L) * (mean_flat50_image.size / np.sqrt(2*np.pi) * std_dark50_stack)\
      * np.exp(-1 * ((center_bins**2) / (2*temporal_variance_dark50_stack)))

print("pdf", pdf.shape, pdf)
"""

plt.figure(8)
plt.title("histogram PRNU")
plt.yscale("log")
plt.ylim(ymin=10**(-1), ymax=10**5)

#histo = np.histogram(highpass.flatten(), bins=Q, range=(highpass.min(), highpass.max()))
plt.hist(highpass.flatten(), bins=2**BIT)
#print("histo[1]", histo[1].shape, histo[0].shape)
#plt.bar(histo[1][:-1], histo[0])
#plt.plot(highpass_hist[0] * histo[0].max())


THRESHOLD = -100
fig = plt.figure(9)
ax = fig.gca()
plt.title("Dead Pixels")
dead_pixel_image = highpass
pos_dead_pixel = np.where(dead_pixel_image < THRESHOLD)
print("Number of dead pixels: ", len(pos_dead_pixel[0]))
print("Dead pixel positions: ", pos_dead_pixel)
#dead_pixel_image[dead_pixel_image > THRESHOLD] = 255
#dead_pixel_image[dead_pixel_image < THRESHOLD] = 0
ax.imshow(dead_pixel_image, cmap=plt.get_cmap("Greys"))
for y,x in zip(*pos_dead_pixel):
    ax.add_patch(Circle((x,y),5,color="r"))


plt.figure(10)
plt.title("histogram DSNU")
plt.yscale("log")
plt.ylim(ymin=10**(-1), ymax=10**5)

plt.hist(mean_dark50_image.flatten(), bins=2**BIT)

THRESHOLD = 550

fig = plt.figure(11)
ax = fig.gca()
plt.title("Hot Pixels")
hot_pixel_image = mean_dark50_image
pos_hot_pixel = np.where(hot_pixel_image > THRESHOLD)
print("Number of hot pixels: ", len(pos_hot_pixel[0]))
print("Hot pixel positions: ", pos_hot_pixel)
#hot_pixel_image[hot_pixel_image > THRESHOLD] = 0
#hot_pixel_image[hot_pixel_image < THRESHOLD] = 255
plt.imshow(hot_pixel_image, cmap=plt.get_cmap("Greys"))
for y,x in zip(*pos_hot_pixel):
    ax.add_patch(Circle((x,y),5, color="r"))


"""FLAT-FIELDING"""
dark_ff = read_dir("flat_fielding/dsnu")
flat_ff = read_dir("flat_fielding/prnu")
test_image = scipy.misc.imread(os.path.join("flat_fielding", "test.png"), mode="I")
test_image = scipy.misc.imresize(test_image, (640,480))
mean_dark_ff_image = np.mean(dark_ff, axis=0)
mean_flat_ff_image = np.mean(flat_ff, axis=0)

print(mean_dark_ff_image.shape, test_image.shape)

plt.figure(12)
plt.subplot(221)
plt.title("test image without correction")
plt.imshow(test_image, cmap=plt.get_cmap("Greys"))


corrected_image = test_image

#corrected_image = flat_fielding(mean_flat_ff_image, mean_dark_ff_image, test_image)
plt.subplot(224)
plt.title("test image corrected")
plt.imshow(corrected_image, cmap=plt.get_cmap("Greys"))




print("done")
plt.show()
