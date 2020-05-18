import pywt
import matplotlib.pyplot as plt
import numpy as np

def plot_wavelet(time, signal, scales, 
                 waveletname = 'morl', 
                 title = 'Wavelet Transform of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname)#, dt)
#     power = (abs(coefficients)) ** 2
#     power = coefficients ** 2
    maximum = 0
    for i in range(len(coefficients)):
        if max(coefficients[i]) > maximum:
            maximum = max(coefficients[i])
    power = abs(coefficients)/maximum

    
    # fig, ax = plt.subplots(figsize=(15, 10))
    # im = ax.contourf(power) #time, frequencies, power) #,cmap=cmap)
    
    # ax.set_title(title, fontsize=20)
    
    # cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    # fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    # plt.show()
    return power

# def get_ave_values(xvalues, yvalues, n = 5):
#     signal_length = len(xvalues)
#     if signal_length % n == 0:
#         padding_length = 0
#     else:
#         padding_length = n - signal_length//n % n
#     xarr = np.array(xvalues)
#     yarr = np.array(yvalues)
#     xarr.resize(signal_length//n, n)
#     yarr.resize(signal_length//n, n)
#     xarr_reshaped = xarr.reshape((-1,n))
#     yarr_reshaped = yarr.reshape((-1,n))
#     x_ave = xarr_reshaped[:,0]
#     y_ave = np.nanmean(yarr_reshaped, axis=1)
#     return x_ave, y_ave

# def plot_signal_plus_average(time, signal, average_over = 5):
#     fig, ax = plt.subplots(figsize=(15, 3))
#     time_ave, signal_ave = get_ave_values(time, signal, average_over)
#     ax.plot(time, signal, label='signal')
#     ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(average_over))
#     ax.set_xlim([time[0], time[-1]])
#     ax.set_ylabel('Signal Amplitude', fontsize=18)
#     ax.set_title('Signal + Time Average', fontsize=18)
#     ax.set_xlabel('Time', fontsize=18)
#     ax.legend()
#     plt.show()