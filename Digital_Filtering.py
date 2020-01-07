# Digital Filtering Code and Example##
# Uses second order section (SOS) butterworth filtering to filter input signals with high
# orders, which avoids numerical issues. Included are functions for low/high pass and band pass filters
#New changes:
from scipy.signal import butter, sosfiltfilt, sosfreqz
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def butter_lowpass(cutoff, fs, order=5):
    # Returns second order section (SOS) filter for low pass filtering
    # The use of SOS provides stable solutions for higher orders by lessening numerical noise(order>2)
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='low', output='sos')
    return sos


def butter_lowpass_filter(data, cutoff, fs, order=5):
    # Digital low pass filter
    sos = butter_lowpass(cutoff, fs, order)
    y = sosfiltfilt(sos, np.ravel(data))
    return y


def butter_highpass(cutoff, fs, order=5):
    # returns second order section (SOS) filter for high pass filtering
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype='high', output='sos')
    return sos


def butter_highpass_filter(data, cutoff, fs, order=5):
    # Digital high pass filter
    sos = butter_highpass(cutoff, fs, order)
    y = sosfiltfilt(sos, np.ravel(data))
    return y


def butter_bandpass(low, high, fs, order=5):
    # returns second order section (SOS) filter for band pass filtering
    nyq = 0.5 * fs
    low_cutoff = low / nyq
    high_cutoff = high / nyq
    sos = butter(order, [low_cutoff, high_cutoff], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, low, high, fs, order=5):
    # Digital band pass filter
    sos = butter_bandpass(low, high, fs, order)
    y = sosfiltfilt(sos, np.ravel(data))
    return y


# Read data with xarray and filter using pandas
ds_disk = xr.open_dataset('W2F6D11_LOWERRING_COLD.nc')
df_disk = ds_disk.to_dataframe()  # convert to dataframe for ease of manipulation
df_disk.reset_index(inplace=True)
df_disk = df_disk.loc[
    (df_disk['Channel'] == "A") & (df_disk['wavelength'].between(1540, 1560, inclusive=True))]  # filter data

#convert to numpy for simplicity
x = df_disk['wavelength'].to_numpy()
y = df_disk['Power (dBm)'].to_numpy()

# Plot raw data
plt.title("Raw Data")
plt.plot(x, y)
plt.show()


### Set up filtering parameters:
##Inputs:
cutoff = 3  # For low/high pass filtering
low, high = 3, 250  # For band pass filtering
order = 5  # Filter order. I.e., steepness of filter around cutoff frequency
##
time_step = (x[-1] - x[0]) / (len(x) - 1)  # Calculate sampling 'time' step from given data
fs = 1 / time_step
###




fig = plt.figure(figsize=(19.20 / 1.2, 10.80 / 1.2))
# Plot the frequency response.

sos = butter_lowpass(cutoff, fs, order)   #sos for low pass filter
# sos = butter_bandpass(low, high, fs, order)   #sos for band pass

plot1 = fig.add_subplot(211)
w, h = sosfreqz(sos, worN=8000)
plt.xscale("log")
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
plt.axvline(cutoff, color='k')
plt.xlim(0, cutoff * 10)
plt.title("Lowpass Filter Frequency Response")
plt.xlabel('Spatial Frequency')
plt.grid()

# Filter the data, and plot both the original and filtered signals.
y_filt = butter_lowpass_filter(y, cutoff, fs, order)  #Low pass filter
# y_filt = butter_bandpass_filter(y, low, high, fs, order)  # band pass filter
plot2 = fig.add_subplot(212)
plt.plot(x, y, 'b-', label='data')
plt.plot(x, y_filt, 'g-', linewidth=2, label='filtered data')
plt.title("Filter & Unfiltered Signal")
plt.xlabel('Wavelength (nm)')
plt.ylabel('dBm')
plt.grid()
plt.legend()
plt.subplots_adjust(hspace=0.35)
plt.show()

#To normalize
#Either divide values by the fit or do a high pass filter instead of the above

##Dividing method:
y_norm = y-y_filt #divide if in linear scale
plt.plot(x, y_norm)
plt.title("Normalized by dividing by the fit")
plt.show()

##Filtering method:
cutoff = 3
order = 5
y_filt_high = butter_highpass_filter(y, cutoff, fs, order)
plt.plot(x, y_filt_high)
plt.title("Normalized by high pass filtering")
plt.show()

#Plot both methods
plt.plot(x, y_norm, label = "Normalized by dividing")
plt.plot(x, y_filt_high, label = " Normalized by filtering")
plt.title("Normalized by dividing and high pass filtering")
plt.legend()
plt.show()

#Filter high freq noise

cutoff = 100
order = 5
y_denoise = butter_lowpass_filter(y_filt_high, cutoff, fs, order)
plt.plot(x, y_filt_high, label='Normalized')
plt.plot(x, y, label='Raw')
plt.plot(x, y_denoise,  label='Normalized, noise filtered')
plt.title("Normalized with and without noise filter")
plt.legend()
plt.show()

