import corr
import time
import struct
import matplotlib.pyplot as plt
import numpy as np

# initial setup
def setupRoach(fpga):
    fpga.progdev('guppi2_1024_t12_w095_p00_s6_2013_Jul_24_1257.bof')
    time.sleep(2)

    fpga.write_int('IP_0', 10*(2**24)+0*(2**16)+0*(2**8)+145)
    fpga.write_int('PT_0', 60000)

    SOURCE_IP=10*(2**24)+0*(2**16)+0*(2**8)+4
    FABRIC_PORT=60000
    MAC_BASE=(2<<32)+(2<<40)

    fpga.tap_start('tap0','tGX8_tGv20',MAC_BASE,SOURCE_IP,FABRIC_PORT)
    time.sleep(1)

    fpga.write_int('FFT_SHIFT', 0xAAAAAAAA)
    fpga.write_int('SCALE_P0', 0x800000)
    fpga.write_int('SCALE_P1', 0x800000)
    fpga.write_int('N_CHAN', 0)

# arm and simulate a PPS so that the board starts transmission
def armAndSimPPS(fpga):
    fpga.write_int('ARM', 0)
    fpga.write_int('PPS_SIM', 0)
    fpga.write_int('ARM', 1)
    fpga.write_int('PPS_SIM', 1)
    fpga.write_int('PPS_SIM', 0)
    fpga.write_int('ARM', 0)

# plot ADC shared BRAMs
def plotADC(fpga):
    # trigger capture
    fpga.write_int('DC_EN', 0)
    fpga.write_int('raw_adc_trig', 0)
    fpga.write_int('DC_EN', 1)
    fpga.write_int('raw_adc_trig', 1)
    fpga.write_int('DC_EN', 0)
    fpga.write_int('raw_adc_trig', 0)

    # read shared BRAMs
    a = struct.unpack('>2048I', fpga.read('P0_DC_SAMP_3_Shared_BRAM', 2048*4))
    b = struct.unpack('>2048I', fpga.read('P0_DC_SAMP_2_Shared_BRAM', 2048*4))
    c = struct.unpack('>2048I', fpga.read('P0_DC_SAMP_1_Shared_BRAM', 2048*4))
    d = struct.unpack('>2048I', fpga.read('P0_DC_SAMP_0_Shared_BRAM', 2048*4))

    # pack data
    adc = [val for tuple in zip(d, c, b, a) for val in tuple]
    adc2 = np.asarray(adc, dtype=np.int32).view(dtype=np.int8)

    # plot
    plt.figure(1)
    plt.plot(adc2)
    plt.show()

if __name__ == '__main__':
    fpga=corr.katcp_wrapper.FpgaClient('192.168.40.99',7147)
    time.sleep(1)

    setupRoach(fpga)
    armAndSimPPS(fpga)

