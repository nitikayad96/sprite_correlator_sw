import sys
import time
import numpy as np
import adc5g as adc
import pylab
import socket
import ami as AMI
import helpers as helpers
import control as control
import file_writer as fw
import pylab
import signal
import logging

logger = helpers.add_default_log_handlers(logging.getLogger(__name__))

def write_data(writer, d, timestamp, meta, **kwargs):
    if meta is not None:
        for entry in meta.entries:
           name = entry['name']
           if name is not 'obs_name':
               val = meta.__getattribute__(name)
               try:
                   length = len(val)
                   data_type = type(val[0])
               except TypeError:
                   length = 1
                   data_type = type(val)
               #print name,val,data_type
               writer.append_data(name, [length], val, data_type)
    writer.append_data('xeng_raw0', d.shape, d, np.int64)
    writer.append_data('timestamp0', [1], timestamp, np.int64)
    for key, value in kwargs.iteritems():
        writer.append_data(key, value.shape, value, value.dtype)

def signal_handler(signum, frame):
    """
    Run when kill signals are caught
    """
    print "Received kill signal %d. Closing files and exiting"%signum
    writer.close_file()
    try:
        ctrl.close_sockets()
    except:
       pass #this is poor form
    exit()


if __name__ == '__main__':
    from optparse import OptionParser

    p = OptionParser()
    p.set_usage('%prog [options] [CONFIG_FILE]')
    p.set_description(__doc__)
    p.add_option('-t', '--test_tx', dest='test_tx',action='store_true', default=False, 
        help='Send tx test patterns, and don\'t bother writing data to file')

    opts, args = p.parse_args(sys.argv[1:])

    if args == []:
        config_file = None
    else:
        config_file = args[0]

    writer = fw.H5Writer(config_file=config_file)
    bl_order = [[0,0], [1,1], [0,1]]
    writer.set_bl_order(bl_order)

    corr = AMI.spriteSbl(config_file=config_file) #passive=True)
    time.sleep(0.1)

    xeng = corr.xengs[0]

    # some initial values for the loop
    cnt=0
    datavec = np.zeros([corr.n_chans*corr.n_bands,corr.n_bls,corr.n_pols,2],dtype=np.int64)
    print('Shape of datavec: ', datavec.shape)
    current_obs = None
    mcnt_old = xeng.read_uint('mcnt_lsb')
    receiver_enable = False
    last_meta_timestamp = time.time()
    # get noise switch data
    noise_switched_data = np.zeros([corr.n_ants, corr.n_chans*corr.n_bands], dtype=np.float32)
    # ignore phasing
    phased_to = np.array([0.0, corr.array.lat_r])

    # Catch keyboard interrupt and kill signals (which are initiated by amisa over ssh)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    while(True):

        if current_obs is None:
            fname = 'corr_TEST_%d.h5'%(time.time())
            writer.start_new_file(fname)
            current_obs = 'test'

        mcnt = xeng.read_uint('mcnt_lsb')
        if mcnt != mcnt_old:
            mcnt_old = mcnt
            d = corr.snap_corr_wide(wait=True,combine_complex=False)
            cnt += 1

            if d is not None:                
                datavec[:,0,0,1] = d['corr00']
                datavec[:,1,0,1] = d['corr11']
                datavec[:,2,0,1] = d['corr01'][0:2048] #datavec[:,:,:,1] should be real
                datavec[:,2,0,0] = d['corr01'][2048:] #datavec[:,:,:,0] should be imag
                print "got new correlator data with timestamp %.4f at time %.4f"%(d['timestamp'], time.time())

                #txdata = np.array(d['corr01'][:], dtype=np.int32)

                write_data(writer,datavec,d['timestamp'],None,noise_demod=noise_switched_data, phased_to=phased_to)
            else:
                print "Failed to send because MCNT changed during snap"

        time.sleep(0.05)
