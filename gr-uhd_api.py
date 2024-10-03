# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16  2024

@author: Tawfik Osman

"""
import gnuradio
from gnuradio import uhd
from gnuradio import blocks
from gnuradio import gr
import time
from signal_processing import *
from dependencies import Estimator


class tx(gr.top_block):

    def __init__(self,device_dict):
        gr.top_block.__init__(self, "TX")

        # Variables
        self.usrp_id =  device_dict['id']
        self.samp_rate = device_dict['sample_rate']
        self.center_freq = device_dict['center_freq']
        self.gain  = device_dict['rf_gain']
        self.clock_mode = device_dict['clock_mode']
        self.rf_channels = device_dict['rf_channels']
        self.channel_type = device_dict['rf_port']
        # Blocks
        self.uhd_stream_arg = uhd.stream_args(cpu_format="fc32",otw_format="sc16",channels=self.rf_channels,)
        self.usrp_tx = uhd.usrp_sink(",".join((f'serial={self.usrp_id}', "")), self.uhd_stream_arg,)


        self.usrp_tx.set_clock_source(f'{self.clock_mode}', 0)
        self.usrp_tx.set_time_source(f'{self.clock_mode}', 0)

        # USRP 1
        for i in self.rf_channels:
            self.usrp_tx.set_time_unknown_pps(uhd.time_spec(0))
            self.usrp_tx.set_time_now(uhd.time_spec(5))
            self.usrp_tx.set_samp_rate(self.samp_rate)
            self.usrp_tx.set_center_freq(self.center_freq, i)
            self.usrp_tx.set_antenna(self.channel_type, i)
            self.usrp_tx.set_normalized_gain(self.gain, i)
            self.usrp_tx.set_bandwidth(self.samp_rate, i)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.usrp_tx.set_samp_rate(self.samp_rate)
        for i in self.rf_channels:
            self.usrp_tx.set_bandwidth(self.samp_rate, i)

    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        for i in self.rf_channels:
            self.usrp_tx.set_center_freq(center_freq, i)

    def begin_transmit(self, packets):
        self.blocks_vector_source = blocks.file_source(gr.sizeof_gr_complex*1, './files_req/OFDM_packet_upsample_x2_qpsk_v3.dat', True, 0, 0)
        #self.blocks_vector_source = blocks.vector_source_c(packets.tolist(), True, 1, [])
        self.connect((self.blocks_vector_source, 0), (self.usrp_tx, 0))
        return 1
    
    def end_transmit(self, packets):
        self.disconnect((self.blocks_vector_source, 0), (self.usrp_tx, 0))
        return 1
        
class rx(gr.top_block):

    def __init__(self,device_dict):
        gr.top_block.__init__(self, "RX")

        # Variables
        self.usrp_id =  device_dict['id']
        self.samp_rate = device_dict['sample_rate']
        self.center_freq = device_dict['center_freq']
        self.gain  = device_dict['rf_gain']
        self.clock_mode = device_dict['clock_mode']
        self.rf_channels = device_dict['rf_channels']
        self.channel_type = device_dict['rf_port']
        self.sink_file = device_dict['sink_file']
        # Blocks
        self.uhd_stream_arg = uhd.stream_args(cpu_format="fc32",otw_format="sc16", channels=self.rf_channels,)
        self.usrp_rx = uhd.usrp_source(",".join((f'serial={self.usrp_id}', '')),self.uhd_stream_arg,)


        # USRP 1
        self.usrp_rx.set_clock_source(f'{self.clock_mode}', 0)
        self.usrp_rx.set_time_source(f'{self.clock_mode}', 0)
        for i in self.rf_channels:
            self.usrp_rx.set_time_unknown_pps(uhd.time_spec(0))
            self.usrp_rx.set_time_now(uhd.time_spec(5))
            self.usrp_rx.set_samp_rate(self.samp_rate)
            self.usrp_rx.set_center_freq(self.center_freq, i)
            self.usrp_rx.set_antenna(self.channel_type, i)
            self.usrp_rx.set_normalized_gain(self.gain, i)
            self.usrp_rx.set_bandwidth(self.samp_rate, i)
            
        self.link_file_sink()

    def link_file_sink(self):
        self.blocks_streams_to_vector= blocks.streams_to_vector(gr.sizeof_gr_complex, len(self.rf_channels))
        self.blocks_file_sink = blocks.file_sink(gr.sizeof_gr_complex*1, f'{self.sink_file}', False)
        self.blocks_file_sink.set_unbuffered(False)
        self.connect((self.usrp_rx, 0), (self.blocks_streams_to_vector, 0))
        self.connect((self.blocks_streams_to_vector, 0),(self.blocks_file_sink, 0))
        
        return 1
    
    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        for i in self.rf_channels:
            self.uhd_usrp_rx.set_samp_rate(self.samp_rate,i)

    def get_center_freq(self):
        return self.center_freq
    
    def set_center_freq(self, center_freq):
        self.center_freq = center_freq
        for i in self.rf_channels:
            self.uhd_usrp_rx.set_center_freq(self.center_freq, i)

    def open_file(self,save_file): # Write stream to file
        self.blocks_file_sink.open(save_file)
        return 1
    
    def close_file(self):  # close file
        self.blocks_file_sink.close()
        return 1
    
def main_txrx(txUSRP=tx, rxUSRP = rx, options=None):
    deviceDict_TX = {'id':'31F3CC3', 'sample_rate': 20e6, 'center_freq':3.5e9,
                  'rf_gain':1,'clock_mode':'internal', 'rf_channels': [0],
                  'rf_port': 'TX/RX'
                  }
    deviceDict_RX = {'id':'31AC578', 'sample_rate': 20e6, 'center_freq':3.5e9,
                  'rf_gain':.8,'clock_mode':'internal', 'rf_channels': [0],
                  'rf_port': 'RX2',
                  'sink_file':'xxx.dat'
                  }
    
    """ deviceDict_TX = {'id':'325FF54', 'sample_rate': 10e6, 'center_freq':10e6,
                  'rf_gain':0.8,'clock_mode':'internal', 'rf_channels': [0],
                  'rf_port': 'AB'
                  }
    deviceDict_RX = {'id':'329BCBB', 'sample_rate': 10e6, 'center_freq':10e6,
                  'rf_gain':0.4,'clock_mode':'internal', 'rf_channels': [0],
                  'rf_port': 'AB',
                  'sink_file':'xxx.dat'
                  } """
    up = 2
    generator_obj = signal_generator(No_SC=64, NoOFDMSymbols=10, ModOrder= 2, factor = up)
    detector_obj = symbol_detector(No_SC=64, NoOFDMSymbols=10, ModOrder= 2, factor = up)
    #channelEstObj = Estimator(numberSubCarriers=64, numberOFDMSymbols=10, modOrder=2)

    # begin transmission(USRP)

    tx_flowgraph = txUSRP(deviceDict_TX)
    rx_flowgraph = rxUSRP(deviceDict_RX)

    Fs = deviceDict_RX['sample_rate']
    plt.figure(figsize=(12,6))
    
    waveform_Ofdm, _,_ = generator_obj.generate_data_OFDM_symbol(no_zeros_to_pad=50)


    for tb,rb in zip([tx_flowgraph], [rx_flowgraph]):
        no_pkts = 0
        # start transmission
        tb.begin_transmit(waveform_Ofdm.reshape(-1,)) 
        tb.start()
        time.sleep(1)
       
        # start recpeption
        rb.start()
        for _ in range(5):
            if no_pkts>100:
                rb.stop()
                rb.wait()    # reception ended
                break

            else:
                plt.cla()
                rb.open_file('./rx_data.dat')
                time.sleep(0.01)
                rb.close_file()
                received_data = np.fromfile('./rx_data.dat',dtype=np.complex64)
                plt.plot(received_data.real)
                plt.show()
            
                rx_pad_zero = detector_obj.decimateSamples(received_data.reshape(1,-1),factor=up)
                lts_corr, no_pkts,payload_start_indices = detector_obj.detectPeaks(rx_pad_zero)
                output_samples, cfo_estimates = detector_obj.estimate_carrier_freq_offset(rx_pad_zero,payload_start_indices[2],1)
                print(f'New: Estimated LTS CFO : {cfo_estimates*(Fs/2)*1e-3} kHz')

                Hest = detector_obj.estimate_channel(output_samples,payload_start_indices[0])
                #plt.title(f'Number of packets: {len(payload_start_indices)}')
                #plt.plot(lts_corr[0:waveform_Ofdm.shape[1]*10])
                #equalized_symbols = detector_obj.equalizeSymbols(output_samples,payload_start_indices[0],H_est)
                
                #ber,err,receivedSymbols,tx_syms,Hest = channelEstObj.rx_process(rx_pad_zero,payload_start_indices[2],Fs)
                equalized_symbols = detector_obj.equalizeSymbols(output_samples,payload_start_indices[0],Hest)
                plt.scatter(equalized_symbols.real,equalized_symbols.imag)
                
                #plt.title(f'Packet index: {payload_start_indices[2]}, BER: {ber}')
                #plt.scatter(receivedSymbols.real,receivedSymbols.imag)
                #plt.scatter(tx_syms.real,tx_syms.imag)
                plt.xlim([-1.2,1.2])
                plt.ylim([-1.2,1.2])
                ''''''



                plt.pause(1)        

        # stop transmission
        tb.stop()
        tb.wait()
        plt.show()

if __name__ == '__main__':
    import os
    os.system('sudo sysctl -w net.core.rmem_max=24912805')
    os.system('sudo sysctl -w net.core.wmem_max=24912805')
    main_txrx()
