#!/usr/bin/python2.6

import corr, pylab, matplotlib, struct, time, numpy

fpga = corr.katcp_wrapper.FpgaClient('192.168.40.61', 7147)
time.sleep(2)

fpga.progdev('spectrometer_2013_May_16_1347.bof')
time.sleep(2)

fpga.write_int('fftshift', 0x3FF)

# read as 8-bit signed chars
x = numpy.fromstring(fpga.snapshot_get('snap_adc0', man_trig=False, man_valid=False)['data'],dtype='>b')
# re-interpret by dividing by 2^7 to vary between [-1.0, 1.0)
x = [float(i) / 2**7 for i in x]
pylab.subplot(221)
pylab.plot(x, '-o')
x_std = numpy.std(x)
x_mean = numpy.mean(x)

f_s = 800
f = [(float(i) * 2 * f_s / 8192) for i in range(-4096, 4096)]
y = abs(numpy.fft.fftshift(numpy.fft.fft(x)))
pylab.subplot(222)
pylab.plot(f, y, '-o')

# read as 8-bit unsigned chars
spectra = numpy.fromstring(fpga.snapshot_get('snap_spectra', man_trig=False, man_valid=False)['data'],dtype='>I4')
# reinterpret by dividing by 2^31
spectra = [float(i) / 2**31 for i in spectra]
pylab.subplot(223)
pylab.plot(spectra, '-o')
subspectra = spectra[0:512]
pylab.subplot(224)
f_s = 800
f = [(float(i) * 2 * f_s / 1024) for i in range(512)]
pylab.plot(f, subspectra, '-o')

pylab.show()

