import sys
import time
import numpy as np
#import adc5g as adc
import pylab
import socket
import ami as AMI
import helpers 
import struct

if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('%prog [options] [CONFIG_FILE]')
    p.set_description(__doc__)
    p.add_option('-p', '--skip_prog', dest='skip_prog',action='store_true', default=False, 
        help='Skip FPGA programming (assumes already programmed).  Default: program the FPGAs')
    p.add_option('-l', '--passive', dest='passive',action='store_true', default=False, 
        help='Use this flag to connect to the roaches without reconfiguring them')
    p.add_option('-s', '--set_phase_switch', dest='phase_switch', type='int', default=-1, 
        help='override the phase switch settings from the config file with this boolean value. 1 for enable, 0 for disable.')
    p.add_option('-a', '--skip_arm', dest='skip_arm',action='store_true', default=False, 
        help='Use this switch to disable sync arm')
    p.add_option('-t', '--tvg', dest='tvg',action='store_true', default=False, 
        help='Use corner turn tvg. Default:False')
    p.add_option('-m', '--manual_sync', dest='manual_sync',action='store_true', default=False, 
        help='Use this flag to issue a manual sync (useful when no PPS is connected). Default: Do not issue sync')
    p.add_option('-P', '--plot', dest='plot',action='store_true', default=False, 
        help='Plot adc and spectra values')

    opts, args = p.parse_args(sys.argv[1:])

    if args == []:
        config_file = None
    else:
        config_file = args[0]

    # construct the correlator object, which will parse the config file and try and connect to
    # the roaches
    # If passive is True, the connections will be made without modifying
    # control software. Otherwise, the connections will be made, the roaches will be programmed and control software will be reset to 0.
    corr = AMI.spriteSbl(config_file=config_file, program=True)
    time.sleep(0.1)

    COARSE_DELAY = 16*10
    if opts.phase_switch == -1:
        #don't override
        corr.set_phase_switches(override=None)
    else:
        corr.set_phase_switches(override=bool(opts.phase_switch))

    corr.all_fengs('set_fft_shift',corr.c_correlator['fft_shift'])
    corr.all_fengs('set_coarse_delay',COARSE_DELAY)

    #corr.fengs[0].set_coarse_delay(COARSE_DELAY)
    #corr.fengs[1].set_coarse_delay(COARSE_DELAY+100)
    corr.all_fengs('tvg_en',corner_turn=opts.tvg)
    corr.all_xengs('set_acc_len')
    if not opts.skip_arm:
        corr.arm_sync(send_sync=opts.manual_sync)

    # Reset status flags, wait a second and print some status messages
    corr.all_fengs('clr_status')
    time.sleep(2)
    corr.all_fengs('print_status')
    
    if opts.plot:
        # snap some data
        pylab.figure()
        n_plots = len(corr.fengs)
        for fn,feng in enumerate(corr.fengs):
            adc = feng.snap('snapshot_adc', man_trig=True, format='b')
            pylab.subplot(n_plots,1,fn)
            pylab.plot(adc)
            pylab.title('ADC values: ROACH %s, ADC %d, (ANT %d, BAND %s)'%(feng.roachhost.host,feng.adc,feng.ant,feng.band))

        # some non-general code to snap from the X-engine
        print 'Snapping data...'
        d = corr.snap_corr()

        print 'Plotting data...'

        pylab.figure()
        pylab.subplot(4,1,1)
        #pylab.plot(corr.fengs[0].gen_freq_scale(),helpers.dbs(d['corr00']))
        pylab.plot(helpers.dbs(d['corr00']))
        pylab.subplot(4,1,2)
        #pylab.plot(corr.fengs[0].gen_freq_scale(),helpers.dbs(d['corr11']))
        pylab.plot(helpers.dbs(d['corr11']))
        pylab.subplot(4,1,3)
        #pylab.plot(corr.fengs[0].gen_freq_scale(),helpers.dbs(np.abs(d['corr01'])))
        pylab.plot(helpers.dbs(np.abs(d['corr01'])))
        pylab.subplot(4,1,4)
        #pylab.plot(corr.fengs[0].gen_freq_scale(),np.unwrap(np.angle(d['corr01'])))
        pylab.plot(np.unwrap(np.angle(d['corr01'])))
        pylab.show()




