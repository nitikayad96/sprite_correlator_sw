import helpers
from termcolor import colored
import def_fstatus
import time, struct, logging
import adc5g as adc
import roach
import numpy as np
import scipy.linalg #for walsh (hadamard) matrices

logger = helpers.add_default_log_handlers(logging.getLogger(__name__))

class Engine(object):
    """
    A class for F/X engines (or some other kind) which live in ROACH firmware.
    The fundamental assumption is that where multiple engines exist on a ROACH,
    each has a unique prefix/suffix to their register names. (Eg, the registers
    all live in some unique subsystem.
    An engine requires a control register, whose value is tracked by this class
    to enable individual bits to be toggled.
    """
    def __init__(self,roachhost,port=7147,boffile=None,ctrl_reg='ctrl',reg_suffix='',reg_prefix='',connect_passively=True,num=0,logger=logger):
        """
        Instantiate an engine which lives on ROACH 'roachhost' who listens on port 'port'.
        All shared memory belonging to this engine has a name beginning with 'reg_prefix'
        and ending in 'reg_suffix'. At least one control register named 'ctrl_reg' (plus pre/suffixes)
        should exist. After configuring these you can write to registers without
        these additions to the register names, allowing multiple engines to live on the same
        ROACH boards transparently.
        If 'connect_passively' is True, the Engine instance will be created and its current control
        software status read, but no changes to the running firmware will be made.
        """
        self._logger = logger.getChild('(%s:%d)'%(roachhost.host,num))
        #self._logger.handlers = logger.handlers

        self.hostname = roachhost.host
        self.roachhost = roach.Roach(self.hostname, port)
        time.sleep(0.02)
        self.ctrl_reg = ctrl_reg
        self.reg_suffix = reg_suffix
        self.reg_prefix = reg_prefix
        self.num = num
        if connect_passively:
            self.get_ctrl_sw()
        else:
            self.initialise_ctrl_sw()

    def initialise_ctrl_sw(self):
        """Initialises the control software register to zero."""
        self.ctrl_sw=0
        self.write_ctrl_sw()

    def write_ctrl_sw(self):
        """
        Write the current value of the ctrl_sw attribute to the host FPGAs control register
        """
        self.write_int(self.ctrl_reg,self.ctrl_sw)

    def ctrl_sw_edge(self, bit):
        """
        Trigger an edge on a given bit of the control software reg.
        I.e., write 0, then 1, then 0
        """
        self.set_ctrl_sw_bits(bit,bit,0)
        self.set_ctrl_sw_bits(bit,bit,1)
        self.set_ctrl_sw_bits(bit,bit,0)
     
    def set_ctrl_sw_bits(self, lsb, msb, val):
        """
        Set bits lsb:msb of the control register to value 'val'.
        Other bits are maintained by the instance, which tracks the current values of the register.
        """
        num_bits = msb-lsb+1
        if val > (2**num_bits - 1):
            print 'ctrl_sw MSB:', msb
            print 'ctrl_sw LSB:', lsb
            print 'ctrl_sw Value:', val
            raise ValueError("ERROR: Attempting to write value to ctrl_sw which exceeds available bit width")
        # Create a mask which has value 0 over the bits to be changed                                     
        mask = (2**32-1) - ((2**num_bits - 1) << lsb)
        # Remove the current value stored in the ctrl_sw bits to be changed
        self.ctrl_sw = self.ctrl_sw & mask
        # Insert the new value
        self.ctrl_sw = self.ctrl_sw + (val << lsb)
        # Write                                                                                           
        self.write_ctrl_sw()
        
    def get_ctrl_sw(self):
        """
        Updates the ctrl_sw attribute with the current value of the ctrl_sw register.
        Useful when you are instantiating an engine but you don't want to reset
        its control register to zero.
        """
        self.ctrl_sw = self.read_uint(self.ctrl_reg)
        return self.ctrl_sw

    def expand_name(self,name=''):
        """
        Expand a register name with the engines string prefix/suffix
        to distinguish between multiple engines
        on the same roach board
        """
        return self.reg_prefix + name + self.reg_suffix

    def contract_name(self,name=''):
        """
        Strip off the suffix/prefix of a register with a given name.
        Useful if you want to get a list of registers present in an engine
        from a listdev() call to the engines host ROACH.
        """
        return name[len(self.reg_prefix):len(name)-len(self.reg_suffix)]

    def write_int(self, dev_name, integer, *args, **kwargs):
        """
        Write an integer to an engine's register names 'dev_name'.
        This is achieved by calling write_int on the Engine's host ROACH
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the write_int call.
        """
        self.roachhost.write_int(self.expand_name(dev_name), integer, **kwargs)

    def read_int(self, dev_name, *args, **kwargs):
        """
        Read an integer from an engine's register names 'dev_name'.
        This is achieved by calling read_int on the Engine's host ROACH
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the read_int call.
        """
        return self.roachhost.read_int(self.expand_name(dev_name), **kwargs)

    def read_uint(self, dev_name, *args, **kwargs):
        """
        Read an unsigned integer from an engine's register names 'dev_name'.
        This is achieved by calling read_uint on the Engine's host ROACH
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the read_uint call.
        """
        return self.roachhost.read_uint(self.expand_name(dev_name), **kwargs)

    def read(self, dev_name, size, *args, **kwargs):
        """
        Read binary data from an engine's register names 'dev_name'.
        This is achieved by calling read on the Engine's host ROACH
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the read call.
        """
        return self.roachhost.read(self.expand_name(dev_name), size, **kwargs)
        
    def write(self, dev_name, data, *args, **kwargs):
        """
        Read binary data from an engine's register names 'dev_name'.
        This is achieved by calling read on the Engine's host ROACH
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the read call.
        """
        self.roachhost.write(self.expand_name(dev_name), data, **kwargs)
    
    def snap(self, dev_name, **kwargs):
        """
        Call snap on an engine's snap block named 'dev_name'.
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the snap call.
        """
        return self.roachhost.snap(self.expand_name(dev_name), **kwargs)

    def snapshot_get(self, dev_name, **kwargs):
        """
        Call snapshot_get on an engine's snap block named 'dev_name'.
        after expanding the register name with any suffix/prefix. Optional
        arguments are passed down to the snapshot_get call.
        """
        return self.roachhost.snapshot_get(self.expand_name(dev_name), **kwargs)

    def listdev(self):
        """
        Return a list of registers associated with an Engine instance.
        This is achieved by calling listdev() on the Engine's host ROACH,
        and then stripping off prefix/suffixes which are unique to this
        particular engine instance.
        """
        dev_list = self.roachhost.listdev()
        dev_list.sort() #alphebetize
        #find the valid devices, which are those which start with the prefix and end with the suffix
        valid_list = []
        for dev in dev_list:
            if dev.startswith(self.reg_prefix) and dev.endswith(self.reg_suffix):
                valid_list.append(self.contract_name(dev))
        return valid_list

class FEngine(Engine):
    """
    A subclass of Engine, encapsulating F-Engine specific properties.
    """
    def __init__(self, roachhost, ctrl_reg='ctrl', connect_passively=False, num=0, **kwargs):
        """
        Instantiate an F-Engine.
        roachhost: A katcp FpgaClient object for the host on which this engine is instantiated
        ctrl_reg: The name of the control register of this engine
        connect_passively: True if you want to instantiate an engine without modifying it's
        current running state. False if you want to reinitialise the control software of this engine.
        config: A dictionary of parameters for this fengine
        """
        # attributize dictionary
        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        if self.band == 'low':
            self.inv_band = False
        elif self.band == 'high':
            self.inv_band = True
        else:
            raise ValueError('FEngine Error: band can only have values "low" or "high"')
        Engine.__init__(self,roachhost,ctrl_reg=ctrl_reg, reg_prefix='feng%s_'%str(self.adc), reg_suffix='', connect_passively=connect_passively, num=num)
        #Engine.__init__(self,roachhost,ctrl_reg=ctrl_reg, reg_prefix='feng_', reg_suffix=str(self.adc), connect_passively=connect_passively)
        # set the default noise seed
        if not connect_passively:
            self.set_adc_noise_tvg_seed()
            self.phase_switch_enable(self.phase_switch)
            self.noise_switch_enable(True)
            self.set_adc_acc_len()
            self.set_fft_acc_len()
            #self.set_ant_id(self.ant)
        # Figure out if this has old of new style autocorr capture blocks
        if 'auto_snap_acc_cnt' in self.listdev():
            self.has_spectra_snap = False
        else:
            self.has_spectra_snap = True

    def set_walsh(self, N, noise, phase, period=5):
        """
        Set the noise and phase walsh functions of the F-Engine.
        N: order of walsh matrix
        noise: index of noise function
        phase: index of phase function
        period: period (2^?), in multiples of 2**13 clockcyles (firmware specific)
                of shortest walsh step. I.e., 2**13 * 2**<period> * N = period of complete
                walsh cycle in FPGA clocks.
        """
        N_round = int(2**(np.ceil(np.log2(N))))
        walsh_matrix = scipy.linalg.hadamard(N_round)
        walsh_matrix[walsh_matrix == -1] = 0
        phase_func = walsh_matrix[phase]
        noise_func = walsh_matrix[noise]

        self._logger.info("Setting ANT %d (%s) phase walshes to %r"%(self.ant, self.band, phase_func))
        self._logger.info("Setting ANT %d (%s) noise walshes to %r"%(self.ant, self.band, noise_func))

        # Stretch each entry out by a factor of 2**<period>
        phase_slow = []
        noise_slow = []
        for i in range(N_round):
            phase_slow += [phase_func[i] for j in range(2**period)]
            noise_slow += [noise_func[i] for j in range(2**period)]
        
        # The counter in the FPGA cycles through 2**12 ram addresses, so repeat
        # the sequence. Since N_round and 2**period are always powers of 2,
        # this is always an integer number of cycles
        phase_slow_rep = np.array(phase_slow * (2**12 / 2**period / N_round))
        noise_slow_rep = np.array(noise_slow * (2**12 / 2**period / N_round))
        
        # pack in firmware defined format and write to fpga
        dat = np.array((phase_slow_rep << 1) + noise_slow_rep, dtype='>B')
        self.write('switch_states', dat.tostring())
        return dat
 

    def set_ant_id(self, ant_id):
        self.write_int('ant_id', ant_id)

    def config_get(self, key):
        if key in self.config.keys():
            return self.config[key]
        elif key in self.global_config.keys():
            return self.global_config[key]
        else:
            raise KeyError('Key %s not in local or global configs!'%key)

    def set_fft_shift(self,shift):
        """
        Write the fft_shift value for this engine
        """
        self.write_int('fft_shift',shift)

    def gen_freq_scale(self):
        """
        Generate the frequency scale corresponding fo the frequencies of each
        channel produced by this engine (in the order they emerge from the engine's
        FFT. Useful for plotting.
        """
        band = np.arange(0,self.adc_clk/2.,self.adc_clk/2./self.n_chans)
        if self.band == 'low':
           rf_band = self.lo_freq - band
        else:
           rf_band = self.lo_freq + band
        return rf_band
    def set_EQ():
        """
        Set the engine EQ coefficients
        """
        raise NotImplementedError
    def set_coarse_delay(self,delay):
        """
        Set the engine's coarse delay (in FPGA clock cycles)
        """
        self.write_int('coarse_delay',delay)
    def reset(self):
        """
        reset the engine using the control register
        """
        self.ctrl_sw_edge(0)
    def man_sync(self):
        """
        Send a manual sync to the engine using the control register
        """
        self.ctrl_sw_edge(1)
    def arm_trigger(self):
        """
        Arm the sync generator using the control register
        """
        self.ctrl_sw_edge(2)
    def clr_status(self):
        """
        Clear the status flags, using the control register
        """
        self.ctrl_sw_edge(3)
    def clr_adc_bad(self):
        """
        Clear the adc clock bad flag, using the control register
        """
        self.ctrl_sw_edge(4)
    def gbe_rst(self):
        """
        Reset the engine's 10GbE outputs, using the control register
        """
        self.ctrl_sw_edge(8)
    def gbe_enable(self,val):
        """
        Set the engine's 10GbE output enable state to bool(val), using the control regiser
        """
        self.set_ctrl_sw_bits(9,9,int(val))
    def fancy_en(self,val):
        """
        Set the fancy led enable mode to bool(val)
        """
        self.set_ctrl_sw_bits(12,12,int(val))
    def adc_protect_disable(self,val):
        """
        Turn off adc protection if val=True. Else turn on.
        """
        self.set_ctrl_sw_bits(13,13,int(val))
    def tvg_en(self,corner_turn=False,packetiser=False,fd_fs=False,adc=False):
        """
        Turn on any test vector generators whose values are 'True'
        Turn off any test vector generators whose values are 'False'
        """
        self.set_ctrl_sw_bits(17,17,int(corner_turn))
        self.set_ctrl_sw_bits(18,18,int(packetiser))
        self.set_ctrl_sw_bits(19,19,int(fd_fs))
        self.set_ctrl_sw_bits(20,20,int(adc))
        self.ctrl_sw_edge(16)
    def set_adc_noise_tvg_seed(self, seed=0xdeadbeef):
        """
        Set the seed for the adc test vector generator.
        Default is 0xdeadbeef.
        """
        self.write_int('noise_seed', seed)
    def phase_switch_enable(self,val):
        """
        Set the phase switch enable state to bool(val)
        """
        self.set_ctrl_sw_bits(21,21,int(val))
    def noise_switch_enable(self, val):
        """
        Enable or disable the noise switching circuitry
        """
        self.set_ctrl_sw_bits(22,22,int(val))
    def set_adc_acc_len(self, val=None):
        if val is None:
            self.write_int('adc_acc_len', self.adc_power_acc_len >> (4 + 8))
        else:
            self.write_int('adc_acc_len', val >> (4 + 8))
    def set_fft_acc_len(self, val=None):
        if 'auto_acc_len' in self.listdev():
            if val is None:
                self.write_int('auto_acc_len', self.fft_power_acc_len)
            else:
                self.write_int('auto_acc_len', val)
        else:
            if val is None:
                self.write_int('auto_acc_len1', self.fft_power_acc_len)
            else:
                self.write_int('auto_acc_len1', val)
    def set_tge_outputs():
        """
        Configure engine's 10GbE outputs.
        Not yet implemented
        """
        raise NotImplementedError()
    def get_status(self):
        """
        return the status flags defined in the def_fstatus file
        """
        val = self.read_int('status')
        rv = {}
        for key in def_fstatus.status.keys():
            item = def_fstatus.status[key]
            rv[key] = helpers.slice(val,item['start_bit'],width=item['width'])
        return rv
    def print_status(self):
        """
        Print the status flags defined in the def_fstatus file, highlighting
        and 'bad' flags.
        """
        print "STATUS of F-Engine %d (Antenna %d %s band) on ROACH %s"%(self.adc,self.ant,self.band,self.roachhost.host)
        vals = self.get_status()
        for key in vals.keys():
            if vals[key] == def_fstatus.status[key]['default']:
                print colored('%15s : %r'%(key,vals[key]), 'green')
            else:
                print colored('%15s : %r'%(key,vals[key]), 'red', attrs=['bold'])


    def calibrate_adc(self):
        """
        Calibrate the ADC associated with this engine, using the adc5g.calibrate_mmcm_phase method.
        """
        # The phase switches must be off for calibration
        self.phase_switch_enable(0)
        self._logger.info('Calibrating ADC link')
        adc.calibrate_all_delays(self.roachhost,self.adc,snaps=[self.expand_name('snapshot_adc')])
        # Set back to user-defined defaults
        self.phase_switch_enable(self.phase_switch)
        #opt,glitches =  adc.calibrate_mmcm_phase(self.roachhost,self.adc,[self.expand_name('snapshot_adc')])
        #print opt
        #print glitches
    def get_adc_power(self):
        init_val = self.read_uint('adc_sum_sq0')
        while (True):
            v = self.read_uint('adc_sum_sq0')
            #print v
            if v != init_val:
                break
            time.sleep(0.01)
        v += (self.read_uint('adc_sum_sq1') << 32)
        if v > (2**63 - 1):
            v -= 2**64
        return np.abs(float(v) / (16 * 256 * (self.adc_power_acc_len >> (8 + 4))))

    def get_spectra(self, *args, **kwargs):
        if self.has_spectra_snap:
            return self.get_spectra_snap(*args, **kwargs)
        else:
            return self.get_spectra_nosnap(*args, **kwargs)

    def set_auto_capture(self, val):
        self.write_int('auto_snap_capture', int(val))

    def wait_for_new_spectra(self, last_spectra=None):
        if last_spectra is None:
            last_spectra = self.read_int('auto_snap_acc_cnt')
        acc_cnt = self.read_int('auto_snap_acc_cnt')
        while acc_cnt == last_spectra:
            time.sleep(0.001)
            acc_cnt = self.read_int('auto_snap_acc_cnt')
        return acc_cnt

    def get_spectra_nosnap(self, autoflip=False, safe=True):
        if safe:
            self.set_auto_capture(True)
 
        acc_cnt = self.read_int('auto_snap_acc_cnt')
        try:
            while acc_cnt == self.acc_cnt:
                time.sleep(0.01)
                acc_cnt = self.read_int('auto_snap_acc_cnt')
        except AttributeError:
            # self.acc_cnt won't exist first time round
            pass

        if safe:
            self.set_auto_capture(False)

        self.acc_cnt = acc_cnt
        d = np.ones(self.n_chans)
        s0 = np.array(struct.unpack('>%dl'%(self.n_chans/2), self.read('auto_snap_bram0', self.n_chans*4/2)))
        s1 = np.array(struct.unpack('>%dl'%(self.n_chans/2), self.read('auto_snap_bram1', self.n_chans*4/2)))
        if (not safe) and (self.read_int('auto_snap_acc_cnt') != self.acc_cnt):
            self._logger.warning('Autocorr snap looks like it changed during read')
        
        d[0::2] = s0[:]
        d[1::2] = s1[:]

        d /= (2**20 * float(self.fft_power_acc_len)) #2**20 for binary point compensation
        if autoflip and self.inv_band:
            d = d[::-1]
        return d

    def get_async_spectra(self, autoflip=False):
        d = np.zeros(self.n_chans)
        s0 = np.array(struct.unpack('>%dl'%(self.n_chans/2), self.read('auto_snap_bram0', self.n_chans*4/2)))
        s1 = np.array(struct.unpack('>%dl'%(self.n_chans/2), self.read('auto_snap_bram1', self.n_chans*4/2)))
        #s0 = self.read('auto_snap_bram0', self.n_chans*4/2)
        #s1 = self.read('auto_snap_bram1', self.n_chans*4/2)
        
        for i in range(4):
            d[i::8]   = s0[i::4]
            d[i+4::8] = s1[i::4]
        
        d /= (2**20 * float(self.fft_power_acc_len)) #2**20 for binary point compensation
        if autoflip and self.inv_band:
            d = d[::-1]
        return d

    def get_spectra_snap(self, autoflip=False):
        d = np.zeros(self.n_chans)
        # arm snap blocks
        # WARNING: we can't gaurantee that they all trigger off the same pulse

        # This loop is a hack to make sure the snaps trigger together. NB: This is important since
        # different bits from the same sample end up in multiple snaps. TODO: fix this in firmware
        sync_ok = False
        while (not sync_ok): 
            for i in range(3):
                self.write_int('auto_snap_%d_ctrl'%i,0)
            for i in range(3):
                self.write_int('auto_snap_%d_ctrl'%i,1)
            sync_ok = True
            for i in range(3):
                # After arming, check no snaps have started taking data
                # If they have, rearm and start again
                if self.read_int('auto_snap_0_status') & (2**31 - 1) != 0:
                    sync_ok = False

        # wait for data to come.
        # NB: there is no timeout condition
        done = False
        while not done:
            status = self.read_int('auto_snap_0_status')
            done = not bool(status & (1<<31))
            nbytes = status & (2**31 - 1)
            time.sleep(0.01)

        # grab data
        s0 = np.array(struct.unpack('>%dH'%(nbytes/2), self.read('auto_snap_0_bram', nbytes)))
        s1 = np.array(struct.unpack('>%dH'%(nbytes/2), self.read('auto_snap_1_bram', nbytes)))
        s2 = np.array(struct.unpack('>%dH'%(nbytes/2), self.read('auto_snap_2_bram', nbytes)))
        s = np.array([s0, s1, s2])

        # each snap is 128 bits wide. 3 snaps is 384 bits -> 8 * 48 bit signed integers
        # Do bit manipulations to carve up the snaps into 16 bit chunks and glue them together appropriately
        for i in range(8):
            top16 = s[(i*3 + 0) // 8, (i*3 + 0) % 8 :: 8]
            mid16 = s[(i*3 + 1) // 8, (i*3 + 1) % 8 :: 8]
            low16 = s[(i*3 + 2) // 8, (i*3 + 2) % 8 :: 8]
            d[i::8] = helpers.uint2int((top16 << 32) + (mid16 << 16) + low16, 48, 34, complex=False)
        
        d /= float(self.fft_power_acc_len)
        if autoflip and self.inv_band:
            d = d[::-1]
        return d

    def get_quant_spectra(self, autoflip=False):
        d = self.snap('quant_snap', format='B')
        # The results are 4bit real/imag pairs, so reformat
        if autoflip and self.inv_band:
            return helpers.uint2int(d, 4, 3, complex=True)
        else:
            return helpers.uint2int(d, 4, 3, complex=True)[::-1]
        
           
    def get_eq(self, redishost=None, autoflip=False, per_channel=False):
        n_eq_coeffs = 2 * self.n_chans / self.eq_dec #2 for complex
        n_bytes_per_coeff = struct.calcsize(self.eq_format)
        n_bytes = n_bytes_per_coeff * n_eq_coeffs
        n_bits_per_coeff = n_bytes*8
        if redishost is not None:
            d = redishost.get('ANT%d_%s'%(self.ant, self.band))
        else:
            d = uint2int(struct.unpack('>%d%s'%(self.n_eq_coeffs, self.eq_format), self.read('eq',n_bytes)), n_bits_per_coeff, self.eq_bp, complex=True)
        if per_channel:
            #expand the coeff array so we have one coeff per channel
            out = np.zeros(self.n_chans, dtype=complex)
            for i in range(self.eq_dec):
                out[i::self.eq_dec] = d
        else:
            out = d
        if autoflip and self.inv_band:
            return out[::-1]
        else:
            return out


class XEngine(Engine):
    """
    A subclass of Engine, encapsulating X-Engine specific properties
    """
    def __init__(self,roachhost, ctrl_reg='ctrl', id=0, connect_passively=True, num=0, **kwargs):
        """
        Instantiate a new X-engine.
        roachhost: The hostname of the ROACH on which this Engine lives
        ctrl_reg: The name of the control register of this engine
        id: The id of this engine, if multiple are present on a ROACH
        connect_passively: True if you want to instantiate an engine without modifying it's
        current running state. False if you want to reinitialise the control software of this engine.
        """
        # attributize dictionary
        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

        Engine.__init__(self,roachhost,ctrl_reg=ctrl_reg,connect_passively=connect_passively,reg_prefix='xeng_', num=num)
        self.id = id
        if not connect_passively:
            self.set_acc_len()

    def reset():
        """
        Reset this engine
        """
        pass
    
    def set_channel_map(self, map):
        self.roachhost.write('packetizer_chan_num', struct.pack('>%dL'%(2*self.n_chans), *map))
        self.chan_map = np.array(map, dtype=int)

    def get_channel_map(self):
        try:
            return self.chan_map
        except AttributeError:
            self._logger.error('Tried to get X-Engine channel map but it hasn\'t been set!')

    def config_output_gbe(self, src_ip, dest_ip, dest_mac, port):
        src_ip_int = helpers.ip_str2int(src_ip) 
        dest_ip_int = helpers.ip_str2int(dest_ip) 
        mac_int = src_ip_int + 0xc0000000
        self.write_int('one_gbe_tx_port', port)
        self.write_int('one_gbe_tx_ip', dest_ip_int)
        self.roachhost.write('one_GbE', struct.pack('>Q', mac_int), offset=0)
        self.roachhost.write('one_GbE', struct.pack('>L', src_ip_int), offset=0x10)
        self.roachhost.write('one_GbE', struct.pack('>Q', dest_mac), offset=(0x3000+8*(dest_ip_int & 0xff)))
        #self.roachhost.tap_start('one_GbE', 'one_GbE', mac_int, src_ip_int, port)

    def set_engine_id(self, id):
        self.write_int('id', id)

    def set_acc_len(self,acc_len=None):
        """
        Set the accumulation length of this engine, using either the
        current value of the acc_len attribute, or a new value if supplied
        """
        if acc_len is not None:
            self.acc_len = acc_len
        # The FPGA expects acc_len - 1 to be written to the 'acc. len maximum index' register
        # self.write_int('acc_len_mi',self.acc_len-1)

	# modified write to register name "acc_len"
	self.write_int('acc_len',self.acc_len-1)

    def set_vacc_arm(self, mcnt):
        """
        Arm the vacc to start recording data at this accumulation.
        """
        self.write_int('target_mcnt', mcnt)

    def reset_vacc(self):
        self.ctrl_sw_edge(0)

    def reset_ctrs(self):
        self.ctrl_sw_edge(3)

    def reset_gbe(self):
        self.ctrl_sw_edge(1)

    def subscribe_mc(self, addr, n_addr=1):
        """
        Subscribe this X-engine to a multicast stream
        """
        for i in range(4):
            a = addr+n_addr*i
            self.roachhost.write_int('network_link%d_core'%(i+1), (addr+n_addr*i), offset=12)
            self.roachhost.write_int('network_link%d_core'%(i+1), 2**32 - (2**n_addr - 1), offset=13)

    def unsubscribe_mc(self):
        """
        Subscribe this X-engine to a multicast stream
        """
        for i in range(4):
            self.roachhost.write_int('network_link%d_core'%(i+1), 0, offset=12)
            self.roachhost.write_int('network_link%d_core'%(i+1), 0, offset=13)
