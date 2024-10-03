import scipy.io as sc
import numpy as np
data = sc.loadmat('OFDM_packet_upsample_x2_16qam_v4.mat')['tx_vec_air_A'].astype(np.complex64)
data[0,:].tofile('OFDM_packet_upsample_x2_16qam_v4.dat')

print(data.shape)
