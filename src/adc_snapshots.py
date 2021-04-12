import serial
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import ami


parser = argparse.ArgumentParser(description='Capture ADC snapshots at different frequencies/powers')
parser.add_argument('-f', '--freqs', type=float, action='append', nargs='+', help='List of frequencies in MHz to set on Valon and grab ADC snapshots at')
parser.add_argument('-a', '--attens', type=float, action='append', nargs='+', help='List of attenuations in dB to set within Valon and grab ADC snapshots at')
parser.add_argument('-p','--plot',type=str, choices=['time','fft'], help="Plot either ADC timeseries or FFT of ADC snapshot")
parser.add_argument('-o','--outfile', type=str, help='Name of outfile of plot to be placed within adc_snapshots directory')

args = parser.parse_args()

freqs = args.freqs[0]
attens = args.attens[0]

corr = ami.spriteSbl(config_file='/home/sprite/installs/sprite_correlator_sw/config/sprite.yaml')
outdir = '/home/sprite/installs/sprite_correlator_sw/adc_snapshots'

v5009 = serial.Serial('/dev/ttyUSB4', baudrate=9600, timeout=1.0)

if not v5009.isOpen():
	v5009.open()

fig = plt.figure(figsize=(10,5))
for a in attens:
	cmd_a = 'source 2; att %2.1f\r' %(a)
	v5009.write(cmd_a)

	for f in freqs:
		
		cmd_f = 'source 2; freq %3.1f\r' %(f)
		v5009.write(cmd_f)
		time.sleep(5)

		adc = []
		for fn, feng in enumerate(corr.fengs):
			adc.append(feng.snap('snapshot_adc', man_trig=True, format='b'))

		if args.plot == 'fft':

			y = (np.fft.fftshift(np.abs((np.fft.fft(adc[0])))**2))[8192:]
			x = np.arange(8192)*2000/8192

			plt.plot(x,y, label=('%3.1f MHz, %2.1f dB' %(f, a)))

		if args.plot == 'time':
			
			plt.plot(adc[0], label=('%3.1f MHz, %2.1f dB' %(f, a)))
			xmax = 10*(2000/f)
			plt.xlim([0,xmax])

		
plt.legend()

plt.savefig(os.path.join(outdir, args.outfile), bbox_inches='tight')

#plt.show()

