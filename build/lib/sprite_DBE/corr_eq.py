import sys
import os
import time
import struct
import numpy as np
import adc5g as adc
import pylab
import socket
import ami as AMI
from helpers import uint2int, dbs
import cPickle as pickle

def calc_eq_factor(d, target_power, snap_bits=18, quant_bits=4):
    bit_diff = snap_bits - quant_bits
    target_bits = target_power + bit_diff
    return np.sqrt(1./d) * 2**target_bits

def format_eq(eq, bits=16, bp=6):
    # convert the EQ into appropriately scaled integers
    # which will be correctly interpretted on the FPGA
    ints = np.round(eq*(2**bp))
    # saturate (coefficients are signed)
    ints[ints>2**(bits-1) - 1] = 2**(bits-1) - 1
    # pack as binary string
    eq_str = ''
    for v in ints:
        eq_str += struct.pack('>h',v)
        eq_str += struct.pack('>h',v)
    return eq_str

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('%prog [options] [CONFIG_FILE]')
    p.set_description(__doc__)
    p.add_option('-N', '--samples', dest='samples', type='int', default=1,
        help='Number of snaps to average over. Default=1')
    p.add_option('-t', '--targetpower', dest='targetpower', type='float', default=0.04,
        help='Mean power target. Default:0.04')
    p.add_option('-c', '--cutoff', dest='cutoff', type='float', default=10.,
        help='Set a cutoff level for <cutoff> EQ coefficients. Where power[channel] < (mean power / cutoff), EQ coefficients will be set to zero. Default: 10.0')
    p.add_option('--same', dest='same', action='store_true', default=False,
        help='Use this flag to load the coefficients from feng0 to all F-engines (can\'t be used in conjunction with --new)')
    p.add_option('-p', '--plot', dest='plot', action='store_true', default=False,
        help='Show plots. Default: False')

    opts, args = p.parse_args(sys.argv[1:])
    load_new = True
    if args == []:
        config_file = None
    else:
        config_file = args[0]

    # initialise connection to correlator
    corr = AMI.spriteSbl(config_file=config_file)
    time.sleep(0.1)

    # make new coefficients always
    coeffs = {}    
    decimation=2
    vec_width = corr.n_chans
    for feng in corr.fengs:
        # turn off noise switch so we can sample powers
        #feng.noise_switch_enable(False)
        time.sleep(1)
        if load_new:
            print "Computing new coefficients"
            d = np.zeros(vec_width)
            for n in range(opts.samples):
                print '%d: Snapping data from ANT: %d, BAND: %s'%(n,feng.ant,feng.band)
                d += feng.get_spectra()
            # calculate target mean power, scaled for 4 bits
            d /= float(opts.samples)
            mean_power = np.mean(d)
            lowlimit = mean_power / float(opts.cutoff)
            if opts.plot:
                pylab.figure(1)
                pylab.plot(dbs(d),label='ANT %d, BAND %s'%(feng.ant,feng.band))
                pylab.axhline(y=10*np.log10(lowlimit),label='ANT %d, BAND %s limit'%(feng.ant,feng.band), color=pylab.gca().lines[-1].get_color())
                pylab.legend()
                pylab.title("Autocorrelation Passbands")
                pylab.ylabel("Power (db)")
                pylab.xlabel("Decimated Channel Number")

            eq = (np.sqrt(1./d)) * opts.targetpower
            #eq[eq>np.mean(eq)*opts.cutoff] = 0
            eq[d<lowlimit] = 0
            eq = eq[::decimation]
            if opts.plot:
                pylab.figure(2)
                pylab.plot(eq,label='ANT %d, BAND %s'%(feng.ant,feng.band))
                pylab.title("EQ coefficients")
                pylab.ylabel("Amplitude (linear)")
                pylab.xlabel("Decimated Channel Number")
                pylab.legend()
            # save eq in a dictionary ready for pickling
            coeffs['ANT%d_%s'%(feng.ant,feng.band)] = eq

        eq_str = format_eq(eq,bits=16,bp=6)
        feng.write('eq', eq_str)
        rb = struct.unpack('>1024H', feng.read('eq',1024*2))

        print "Grabbing snapshot of quantized signal for Antenna %d %s band"%(feng.ant,feng.band)

        quant = uint2int(feng.snap('quant_snap',format='B',wait_period=3),4,3,complex=True)[0:corr.n_chans]
        print 'done'
        dev = np.std(np.abs(quant))
        levelwidth = 2**-3
        print 'level width = %.3f, standard deviation of amplitude: %.3f' %(levelwidth, dev)
        print 'E = %.3f x dev:' %(levelwidth/dev)
        if opts.plot:
            pylab.figure(4)
            pylab.plot(np.imag(quant),label='ANT %d, BAND %s'%(feng.ant,feng.band))
            pylab.title("Quantized signal (real part) (normalised to 1)")
            pylab.ylabel("Amplitude (linear)")
            pylab.xlabel("Channel Number")
            pylab.legend()


    if opts.plot:
        pylab.show()
