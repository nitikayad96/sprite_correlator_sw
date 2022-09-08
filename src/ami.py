import os
import numpy as np
import time
import struct
import yaml
import helpers
import threading
import Queue
import logging
import roach
import engines
import antenna_functions
from corr import sim
from fractions import gcd

logger = helpers.add_default_log_handlers(logging.getLogger(__name__))

def _queue_instance_method(q, num, inst, method, args, kwargs):
    '''
    Add an [num, inst.method(*args, **kwargs)] call to queue, q.
    Use q.get() to get the return data from this call.
    
    This function is used for parallelizing calls to multiple
    roaches/engines.
    '''
    q.put([num, getattr(inst, method)(*args, **kwargs)])


class spriteDBE(object):
    """
    The sprite roach Digital Correlator class.
    This class provides an interface to the correlator, using a config file
    for hardware set up.
    """
    def __init__(self, config_file='/home/sprite/installs/sprite_correlator_sw/config/sprite.yaml', logger=logger, program=False):
        """
        Instantiate a correlator object. Pass a config file for custom configurations.
        """
        self._logger = logger

        # TODO add exception for no config file
    
        self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        self._logger.info("Building spriteDBE with config file %s"%config_file)
        self._logger.info("parsing config file")
        self.parse_config_file()
        self._logger.info("connecting to roaches")
        self.connect_to_roaches()
        time.sleep(0.1)

        # reprogramming messes with ctrl_sw, etc, so clean out the engine lists
        #self._logger.info("Re-initializing engines")
        self.initialise_f_engines(passive=False)
        self.initialise_x_engines(passive=False)

        
        #if program:
        #    self.program_all()

    def set_walsh(self, period=5, noise=15):
        for feng in self.fengs:
            dat = feng.set_walsh(2*self.n_ants, noise, feng.phase_walsh, period)
            # Also write to the same roach's GPIO outputs.
            feng.roachhost.write('gpio_switch_states', dat.tostring())

    def report_alive(self, name, arguments=[], timeout=5):
        self.redis_host.set(name+'_ALIVE', arguments, ex=timeout)

    def set_phase_switches(self, override=None):
        '''
        Override phase switches enable with specified state.
        If no state is given, set to the default enabled
        state from the config file.
        '''
        for fn, feng in enumerate(self.fengs):
            if override is not None:
                self._logger.warning('Overriding phase switch enable state of feng %d to %s'%(fn, override))
                feng.phase_switch_enable(override)
            else:
                self._logger.info('Setting phase switch enable state of feng %d to %s'%(fn, feng.phase_switch))
                feng.phase_switch_enable(feng.phase_switch)


    def set_ip_base(self, ip):
        self._logger.info('Setting base IP to %s'%ip)
        ip_list = map(int, ip.split('.'))
        # We only write the top 24 bits to the FPGA
        ip_int = (ip_list[0] << 16) + (ip_list[1] << 8) + ip_list[2]
        for host, fpga in self.fpgas.iteritems():
            fpga.write_int('ip_base', ip_int)

    def get_mcnt2time_factors(self):
        """
        Get the offset and mult fact required to turn an mcnt into a time
        """
        conv_factor = self.c_correlator_hard['window_len']/(self.adc_clk)
        offset = self.load_sync()
        return {'offset': offset, 'conv_factor': conv_factor}

    def mcnt2time(self,mcnt):
        """
        Convert an mcnt to a UTC time based on the instance's sync_time attribute.
        """
        conv_factor = self.fengs[0].n_chans * 2 * self.c_correlator_hard['window_len']/(self.adc_clk)
        offset = self.load_sync()
        return offset + mcnt * conv_factor

    def time_to_mcnt(self, time):
        mcnt_zero = self.load_sync()
        if time < mcnt_zero:
            self._logger.error('Requested time %f, which is prior to current sync time %f'%(time, offset))
            raise RuntimeError('Requested time %f, which is prior to current sync time %f'%(time, offset))
        offset = time - mcnt_zero
        conv_factor = self.fengs[0].n_chans * 2 * self.c_correlator_hard['window_len']/(self.adc_clk)
        return int(offset / conv_factor)
    

    def round_time_to_sync_clocks(self, time):
        """
        Given a target time, return the number of adc clocks 
        of the time rounded up so that it coincides with an FPGA sync.
        """
        # get the current sync time
        synctime = self.load_sync()
        # get the time since sync of our target time
        elapsed = time - synctime
        # how many syncs is this?
        ##print 'elapsed', elapsed
        ##print 'sync time', self.sync_period
        n_syncs = elapsed / self.sync_period
        ##print 'n_syncs', n_syncs
        # round up and convert to adc clocks
        target_sync = int(np.ceil(n_syncs))
        ##print 'taget sync', target_sync
        ##print 'sync_clocks', self.sync_clocks
        ##print 'rv', target_sync * self.sync_clocks
        return target_sync * self.sync_clocks

    def arm_vaccs(self):

        for xeng in self.xengs:
            mcnt_msb = xeng.read_uint('mcnt_msb')
            mcnt_lsb = xeng.read_uint('mcnt_lsb')
            mcnt = (mcnt_msb << 32) + mcnt_lsb
            arm_mcnt = (mcnt_msb+1 << 32) + mcnt_lsb 
            xeng.set_vacc_arm(arm_mcnt & (2**32-1)) # we only trigger on the bottom 32 bits
            xeng.reset_vacc()


    def get_current_mcnt(self):
        return self.time_to_mcnt(time.time())

    def arm_sync(self, send_sync=False):
        """
        Arms all F engines, records anticipated sync time in config file. Returns the UTC time at which
        the system was sync'd in seconds since the Unix epoch (MCNT=0)
        If send_sync = True, a manual software sync will be sent to the F-Engines after arming.
        """
        #wait for within 100ms of a half-second, then send out the arm signal.
        #TODO This code is ripped from medicina. Is it even right?
        
        #self._logger.info('Issuing F-Engine arm')
        #ready=0
        #while not ready:
        #    ready=((int(time.time()*10+5)%10)==0)
        if not send_sync:
            t = self.fengs[0].roachhost.read_int('pps_count')
            while self.fengs[0].roachhost.read_int('pps_count') == t:
                time.sleep(0.001)
        #+4 because the firmware is configured to sync after 4 PPS pulses
        #+1 because there is a bug in the sync block!!! so sync comes one pps late(!)
        self.sync_time=int(time.time())+4+1
        self._logger.info("Arming F-engine syncs at time %.3f"%self.sync_time)
        self.all_fengs('arm_trigger')
        if send_sync:
            # send two syncs, as the first is flushed
            self.all_fengs('man_sync')
            self.all_fengs('man_sync')
        #self.redis_host.set('sync_time', self.sync_time)
        time.sleep(4+1+1) #wait one second more than necessary for a sync
        return self.sync_time

    def load_sync(self):
        """Determines if a sync_time exists, returns that value else return 0"""
        t = self.redis_host.get('sync_time')
        if t is None:
            self._logger.warning("No previous Sync Time found, defaulting to 0 seconds since the Unix Epoch")
            self.sync_time = 0
        else:
            self.sync_time = t
            self._logger.info('sync time is %d (%s)'%(self.sync_time, time.ctime(self.sync_time)))
        return self.sync_time

    def get_next_sync_acc_alignment(self):
        """
        Return the time of the next sync pulse
        which is aligned with a new accumulation.
        This allows loading of coarse delays at opportune
        moments. Note that if the firmware was cleverer,
        we could load the delays at any software-specified
        time.
        """
        # Get the latest mcnt to which vaccs are currently locked and the sync time
        vacc_mcnt = self.redis_host.get('vacc_arm_mcnt') 
        vacc_time = self.mcnt2time(vacc_mcnt)
        # how long after this time are we?
        # +2 means load will occur at least 2 seconds in the future
        elapsed = (time.time()+2) - vacc_time
        # get lcm of sync period and accumulation length
        lcm = self.sync_clocks * self.acc_samples / gcd(self.sync_clocks, self.acc_samples)
        # how many lcms have passed?
        elapsed_lcms = float(elapsed * self.adc_clk) / lcm
        # round up, and calculate time of next sync / acc-start
        next_match = vacc_time + (np.ceil(elapsed_lcms) * lcm / (self.adc_clk))
        # This is the time we need to have loaded the next coarse delays by.
        return next_match

    def get_source_delays(self, adc_clks=False):
        """
        Return the latest per-antenna delays from redis.
        if adc_clocks = True, return delays in integral adc clocks.
        else, return delays in ps.
        """
        try:
            age = self.redis_host.get_age('CONTROL')
            if age > 10:
                logger.warning('Obtained delays from redis but they are %d seconds old'%age)
            if adc_clks:
                return np.array(1e-12 * self.adc_clk * np.array(self.redis_host.get('CONTROL')['delay']), dtype=int)
            else:
                return np.array(self.redis_host.get('CONTROL')['delay'])
        except TypeError:
            logger.warning('Couldn\'t read source delays from Redis. Using zero delays')
            return np.zeros(self.n_ants)

    def update_coarse_delays(self, delays=None):
        if delays is None:
            delays = [0]*self.n_ants
        if len(delays) != self.n_ants:
            logger.error('Trying to load %d antennas with %d delays!'%(len(delays), self.n_ants))
        for feng in self.fengs:
            feng.set_coarse_delay(delays[feng.ant])
        self.redis_host.set('coarse_delays', list(delays))

    def get_coarse_delay_load_time(self):
        return self.redis_host.get('coarse_delays_valid_at')

    def get_coarse_delays(self):
        return self.redis_host.get('coarse_delays')

    def timed_coarse_delay_update(self, delays=None):
        if delays is None:
            delays = [0]*self.n_ants
        load_by = self.get_next_sync_acc_alignment()
        load_after = load_by - self.sync_period
        while(time.time() < (load_after + 0.3*self.sync_period)):
            time.sleep(0.01)
        startloadtime = time.time()
        logger.info('Delays loading at %s (%.2fs after last sync)'%(time.ctime(startloadtime), startloadtime-load_after))
        self.update_coarse_delays(delays)
        self.redis_host.set('coarse_delays_valid_at', load_by)
        # Warn if loading wasn't done in time
        if time.time() > load_by:
            logger.warning('Coarse delays not loaded in time!')
        # Return load time
        loadtime = time.time()
        logger.info('Delays loaded by %s (%.2fs before next sync)'%(time.ctime(loadtime), load_by-loadtime))
        return load_by

    def parse_config_file(self):
        """
        Parse the instance's config file, saving some parameters as attributes
        for easy access.
        This method is automatically called on object instantiation.
        """
        #some common params
        self.n_ants  = 2
        self.n_bands = 1
        self.n_inputs= 2
        self.n_chans = self.config['Configuration']['correlator']['hardcoded']['n_chans']
        self.n_pols  = self.config['Configuration']['correlator']['hardcoded']['n_pols']
        self.output_format  = self.config['Configuration']['correlator']['hardcoded']['output_format']
        self.data_path  = self.config['Configuration']['correlator']['runtime']['data_path']
        self.adc_clk  = float(self.config['FEngine']['adc_clk'] * 1e6)
        self.lo_freq  = self.config['FEngine']['mix_freq']
        self.f_n_chans  = self.config['FEngine']['n_chans']
        self.x_acc_len = self.config['Configuration']['correlator']['runtime']['acc_len']
        self.n_bls = 4
        
        self.roaches = set([node['host'] for node in self.config['FEngine']['nodes']+self.config['XEngine']['nodes']])

        #shortcuts to sections
        self.c_correlator = self.config['Configuration']['correlator']['runtime']
        self.c_correlator_hard = self.config['Configuration']['correlator']['hardcoded']
        self.c_global = self.config['Configuration']
        self.c_antennas = self.config['Antennas']
        self.c_array = self.config['Array']

        # other useful vars
        self.window_len = self.c_correlator_hard['window_len']
        self.sync_clocks= self.c_correlator_hard['sync_period']*16 #measured in ADC clocks
        self.sync_period = self.sync_clocks / self.adc_clk
        self.acc_samples = self.x_acc_len * self.c_correlator_hard['window_len'] * self.f_n_chans * 2
        self.acc_time = self.acc_samples / self.adc_clk

        # some debugging / info
        self._logger.info("ROACHes are %r"%self.roaches)
        
        # ant array shortcuts
        self.array_lon = self.c_array['lon']
        self.array_lat = self.c_array['lat']
        self.ant_locs = [[0., 0., 0.] for a in self.c_antennas]
        for ant in self.c_antennas:
            self.ant_locs[ant['ant']] = ant['loc'] #this way the ants don't have to be in order in the config file

        self.array = antenna_functions.AntArray((self.array_lat, self.array_lon), self.ant_locs)


    def connect_to_roaches(self):
        """
        Set up katcp connections to all ROACHes in the correlator config file.
        Add the FpgaClient instances to the self.fpgas list.
        Returns a list of boolean values, corresponding to the connection
        state of each ROACH in the system.
        """
        self.fpgas = {}
        connected = {}
        for rn,roachhost in enumerate(self.roaches):
            self._logger.info("Connecting to ROACH %d, %s"%(rn,roachhost))
            self.fpgas[roachhost] = roach.Roach(roachhost,boffile=self.c_global['boffile'])
            time.sleep(0.01)
            connected[roachhost] = self.fpgas[roachhost].is_connected()
            if not connected[roachhost]:
                self._logger.error('Could not connect to roachhost: %s'%roachhost)
        return connected

    def initialise_f_engines(self,passive=False):
        """
        Instantiate an FEngine instance for each one specified in the correlator config file.
        Append the FEngine instances to the self.fengs list.
        """
        self.fengs = []
        for fn, feng in enumerate(self.config['FEngine']['nodes']):
            feng_attrs = {}
            for key in feng.keys():
                feng_attrs[key] = feng[key]
            for key in self.config['FEngine'].keys():
                if (key != 'nodes') and (key not in feng_attrs.keys()):
                    feng_attrs[key] = self.config['FEngine'][key]
            self._logger.info('Constructing F-engine %d (roach: %s, ant: %d, band: %s)'%(fn, feng['host'], feng['ant'], feng['band']))
            self.fengs.append(engines.FEngine(self.fpgas[feng['host']], 
                                      connect_passively=True,
                                      num=fn, **feng_attrs))
        self.n_fengs = len(self.fengs)
        self._logger.info('%d F-engines constructed'%self.n_fengs)

    def noise_switched_from_redis(self):
        noise_switched_data = np.zeros([self.n_ants, self.fengs[0].n_chans*self.n_bands], dtype=np.float32)
        for fn, feng in enumerate(self.fengs):
            ant_index = self.config['Antennas'][feng.ant]['index']
            from_redis = self.redis_host.get('STATUS:noise_demod:ANT%d_%s'%(feng.ant, feng.band))
            if from_redis is not None:
                if feng.band == 'high':
                    noise_switched_data[ant_index,feng.n_chans:2*feng.n_chans] = from_redis
                elif feng.band == 'low':
                    noise_switched_data[ant_index,0:feng.n_chans] = from_redis
            else:
                logger.warning('Couldn\'t get Redis key STATUS:noise_demod:ANT%d_%s'%(feng.ant, feng.band))
        return noise_switched_data


    def enable_debug_mac(self, debug_mac):
        for xn, xeng in enumerate(self.xengs):
            for i in range(4):
                mac_str = struct.pack('>Q', debug_mac)
                xeng.roachhost.write('network_link%d_core'%(i+1), mac_str, offset=(0x3000+8*1))

#    def disable_debug_mac(self):
#        for xn, xeng in enumerate(self.xengs):
#            for i in range(4):
#                mac_str = struct.pack('>Q', 0x02000000 + helpers.ip_str2int(self.c_correlator['ten_gbe']['network'])+ 1)
#                xeng.roachhost.write('network_link%d_core'%(i+1), mac_str, offset=(0x3000+8*1))

#    def tap_channel(self, channel, dest_ip, dest_mac):
#        self._logger.info('Tapping Channel %d -> IP %s, mac 0x%8x'%(channel, dest_ip, dest_mac))
#        dest_ip_int = helpers.ip_str2int(dest_ip)
#        base_ip_int = helpers.ip_str2int(self.c_correlator['ten_gbe']['network'])
#        if (dest_ip_int & 0xffffff00) != (base_ip_int & 0xffffff00):
#            self._logger.error('Debug ip %s and base ip %s are incompatible'%(dest_ip, self.c_correlator['ten_gbe']['network']))

#        # update the relevant mac address in the roach's arp table
#        mac_str = struct.pack('>Q', dest_mac)
#        ram_offset = 0x3000 + (8*(dest_ip_int & 0xff))
#        for host, fpga in self.fpgas.iteritems():
#            for i in range(4):
#                fpga.write('network_link%d_core'%(i+1), mac_str, offset=ram_offset)

#        # now change the destination address of the relevant channel
#        for feng in self.fengs:
#            ram = ''
#            if (feng.band == 'low') and (channel < feng.n_chans):
#                if channel & 1 == 0:
#                    ram = 'network_masker0_params'
#                else:
#                    ram = 'network_masker1_params'
#            if (feng.band == 'high') and (channel >= feng.n_chans):
#                if channel & 1 == 0:
#                    ram = 'network_masker2_params'
#                else:
#                    ram = 'network_masker3_params'
#            if ram != '':
#                ram_loc = channel // 2
#                prev_flags = feng.roachhost.read_int(ram, offset=ram_loc) #offset counts in 32bit words
#                new_flags = (prev_flags & 0xffff0000) + (dest_ip_int & 0xff)
#                feng.roachhost.write_int(ram, new_flags, offset=ram_loc)

#    def start_tge_taps(self):
#        """
#        Start TGE taps on all the boards.
#        Give each core an IP address dependent on the band which the associated
#        X-engine will process.
#        """
#        base_ip_int = helpers.ip_str2int(self.c_correlator['ten_gbe']['network'])
#        for xn, xeng in enumerate(self.xengs):
#            xeng.board_ip = base_ip_int + 4*xn #board level IP. ports have this address + {0..3}
#            for i in range(4):
#                self._logger.info('Starting TGE tap core%d on %s'%(i, xeng.hostname))
#                ip = xeng.board_ip + i
#                mac = 0x02000000 + ip
#                port = 10000
#                self._logger.info('mac: 0x%x, ip: %s, port: %d'%(mac, helpers.ip_int2str(ip), port))
#                mac_table = [0x2000000 + (base_ip_int & 0xffffff00) + m for m in range(256)]
#                xeng.roachhost.config_10gbe_core('network_link%d_core'%(i+1), mac, ip, port, mac_table)

#        for n, ip in enumerate(self.band2ip):
#            self._logger.info('band to ip mapping: band %d -> ip %s'%(n, helpers.ip_int2str(ip)))
                

    def enable_tge_output(self):
        self._logger.info("Enabling TGE outputs")
        [feng.gbe_enable(True) for feng in self.fengs]

    def disable_tge_output(self):
        self._logger.info("Disabling TGE outputs")
        [feng.gbe_enable(False) for feng in self.fengs]

    def reset_tge_flags(self):
        for fn, feng in enumerate(self.fengs):
            if feng.adc == 0:
                feng.roachhost.write('network_masker0_params', np.zeros(feng.n_chans/2, dtype=np.uint32).tostring())
                feng.roachhost.write('network_masker1_params', np.zeros(feng.n_chans/2, dtype=np.uint32).tostring())
            elif feng.adc == 1:
                feng.roachhost.write('network_masker2_params', np.zeros(feng.n_chans/2, dtype=np.uint32).tostring())
                feng.roachhost.write('network_masker3_params', np.zeros(feng.n_chans/2, dtype=np.uint32).tostring())

#    def set_chan_dests(self, enable_output=0, debug_mac=None):
#        """
#        Set the destination IPs / loopback/tge enables
#        for the output data streams
#        """
#        mc_order = self.c_correlator['ten_gbe']['multicast_order']
#        for fn, feng in enumerate(self.fengs):
#            this_board = feng.hostname
#            for xn, xeng in enumerate(self.xengs):
#                if xeng.hostname == this_board:
#                    my_xeng = self.xengs[xn] #The xengine associated with this fengine's board
#            chan_flags = np.arange(feng.n_chans / 2) #Number of channels per link
#            dest_bands = chan_flags % self.n_xengs  #The band each channel belongs to 
#            #dest_ip_base = self.band2ip[dest_bands] & 0xff #The board-level IP (shift up 2 bits and add 0->3 for port level)
#            dest_ip_base = self.band2ip[dest_bands] & 0xfffffffc #The board-level IP add 0->3 for port level)
#            my_ip_base = my_xeng.rcvr_ip & 0xfffffffc
#            

#            lb_vld = np.array(dest_ip_base == my_ip_base, dtype=int)
#            tge_vld = np.array(lb_vld==0, dtype=int) * int(enable_output)
#            #lb_vld = np.zeros_like(tge_vld)
#            # Exclude the ends of the band, which don't go anywhere
#            lb_vld[my_xeng.n_chans*self.n_xengs/2:] = 0
#            tge_vld[my_xeng.n_chans*self.n_xengs/2:] = 0
#            # Put a sync on the first channel of each X-engine for antenna 0
#            sync= np.zeros_like(chan_flags)
#            if feng.ant == 0:
#                sync[0:self.n_xengs] = 1
#            # Every n_xengs (valid) channels we should increment the buffer ID
#            eob = np.zeros_like(chan_flags)
#            eob[self.n_xengs-1:my_xeng.n_chans*self.n_xengs/2:self.n_xengs] = 1
#            eob[my_xeng.n_chans*self.n_xengs:] = 0
#            #for i in range(20):
#            #    print i, helpers.ip_int2str(dest_ip_base[i]), 'tge_vld:', tge_vld[i]==1, 'eob:', eob[i], 'sync:', sync[i]

#            # make a list of channels for this xengine
#            this_xeng_chans = chan_flags[(dest_ip_base == my_ip_base) & ((lb_vld == 1) | (tge_vld == 1))] 
#            chans0 = this_xeng_chans * 2
#            chans1 = this_xeng_chans * 2 + 1
#            chans2 = this_xeng_chans * 2 + feng.n_chans
#            chans3 = this_xeng_chans * 2 + feng.n_chans + 1
#            comp_chans = []
#            for i in range(len(chans0)): #chans0,1,2,3 are all the same length
#                comp_chans += [chans0[i]]
#                comp_chans += [chans1[i]]
#                comp_chans += [chans2[i]]
#                comp_chans += [chans3[i]]
#            my_xeng.set_channel_map(comp_chans)
#            self.redis_host.set('XENG%d_CHANNEL_MAP'%my_xeng.band, comp_chans)
#            self._logger.debug('Xeng %d has channel map %r'%(my_xeng.num, comp_chans))
#            
#            if feng.adc == 0:
#                flags = ((dest_ip_base & 0xff) + 0 * (mc_order or 1)) + (sync<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
#                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
#                feng.roachhost.write('network_masker0_params', flags_str)

#                flags = ((dest_ip_base & 0xff) + 1 * (mc_order or 1)) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
#                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
#                feng.roachhost.write('network_masker1_params', flags_str)
#            elif feng.adc == 1:
#                flags = ((dest_ip_base & 0xff) + 2 * (mc_order or 1)) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
#                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
#                feng.roachhost.write('network_masker2_params', flags_str)

#                flags = ((dest_ip_base & 0xff) + 3 * (mc_order or 1)) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
#                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
#                feng.roachhost.write('network_masker3_params', flags_str)

    def set_chan_dests_half_rate(self, enable_output=0, debug_mac=None):
        """
        Set the destination IPs / loopback/tge enables
        for the output data streams
        """
        for fn, feng in enumerate(self.fengs):
            this_board = feng.hostname
            for xn, xeng in enumerate(self.xengs):
                if xeng.hostname == this_board:
                    my_xeng = self.xengs[xn] #The xengine associated with this fengine's board
            chan_flags = np.arange(feng.n_chans / 2) #Number of channels per link
            dest_bands = (chan_flags >> 1) % self.n_xengs  #The band each channel belongs to (send two consecutive chans to each board, we will only validate the first)
            #dest_ip_base = self.band2ip[dest_bands] & 0xff #The board-level IP (shift up 2 bits and add 0->3 for port level)
            dest_ip_base = self.band2ip[dest_bands] & 0xfffffffc #The board-level IP add 0->3 for port level)
            my_ip_base = my_xeng.rcvr_ip & 0xfffffffc

            lb_vld = np.array(dest_ip_base == my_ip_base, dtype=int)
            tge_vld = np.array(lb_vld==0, dtype=int) * int(enable_output)
            #lb_vld = np.zeros_like(tge_vld)
            # Exclude the ends of the band, which don't go anywhere
            lb_vld[my_xeng.n_chans*self.n_xengs/2:] = 0
            tge_vld[my_xeng.n_chans*self.n_xengs/2:] = 0
            #wipe out every second channel
            lb_vld[::2] = 0
            tge_vld[::2] = 0
            # Put a sync on the first channel of each X-engine
            sync= np.zeros_like(chan_flags)
            sync[0:self.n_xengs] = 1
            # Every 2*n_xengs (valid) channels we should increment the buffer ID
            eob = np.zeros_like(chan_flags)
            eob[self.n_xengs-1:my_xeng.n_chans*self.n_xengs/2:2*self.n_xengs] = 1
            eob[my_xeng.n_chans*self.n_xengs:] = 0
            #for i in range(20):
            #    print i, helpers.ip_int2str(dest_ip_base[i]), 'tge_vld:', tge_vld[i]==1, 'eob:', eob[i], 'sync:', sync[i]

            if feng.adc == 0:
                flags = ((dest_ip_base & 0xff) + 0) + (sync<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
                feng.roachhost.write('network_masker0_params', flags_str)

                flags = ((dest_ip_base & 0xff) + 1) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
                feng.roachhost.write('network_masker1_params', flags_str)
            elif feng.adc == 1:
                flags = ((dest_ip_base & 0xff) + 2) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
                feng.roachhost.write('network_masker2_params', flags_str)

                flags = ((dest_ip_base & 0xff) + 3) + (0<<16) + (tge_vld<<17) + (lb_vld<<18) + (eob<<19)
                flags_str = np.array(flags, dtype=np.uint32).byteswap().tostring()
                feng.roachhost.write('network_masker3_params', flags_str)
                
    def initialise_x_engines(self,passive=False):
        """
        Instantiate an XEngine instance for each one specified in the correlator config file.
        Append the XEngine instances to the self.xengs list.
        """
        self.xengs = []
        for xn, xeng in enumerate(self.config['XEngine']['nodes']):
            xeng_attrs = {}
            for key in xeng.keys():
                xeng_attrs[key] = xeng[key]
            for key in self.config['XEngine'].keys():
                if (key != 'nodes') and (key not in xeng_attrs.keys()):
                    xeng_attrs[key] = self.config['XEngine'][key]
            self._logger.info('Constructing X-engine %d (roach: %s)'%(xn, xeng['host']))
            self.xengs.append(engines.XEngine(self.fpgas[xeng['host']], 'ctrl', connect_passively=True, num=xn, **xeng_attrs))
        self.n_xengs = len(self.xengs)
        self._logger.info('%d X-engines constructed'%self.n_xengs)

#        self.band2ip = np.zeros(self.n_xengs, dtype=int)
#        base_ip_int = helpers.ip_str2int(self.c_correlator['ten_gbe']['network'])
#        mc_order = self.c_correlator['ten_gbe']['multicast_order']
#        for xn, xeng in enumerate(self.xengs):
#            if mc_order != 0:
#                xeng.rcvr_ip = (224<<24) + (0<<16) + (2<<8) + 128 + 4*mc_order*xn
#                if not passive:
#                    xeng.subscribe_mc(xeng.rcvr_ip, mc_order)
#            else:
#	        xeng.rcvr_ip = base_ip_int + 4*xn #board level IP. ports have this address + {0..3}
#            self.band2ip[xeng.band] = xeng.rcvr_ip

    def set_xeng_outputs(self):
        dest_ip = self.c_correlator['one_gbe']['dest_ip']
        dest_mac = self.c_correlator['one_gbe']['dest_mac']
        port = self.c_correlator['one_gbe']['port']
        src_ip_base = ''
        for i in range(3):
            src_ip_base += '%s.'%dest_ip.split('.')[i]

        for xeng in self.xengs:
            xeng.set_engine_id(xeng.band)
            xeng.config_output_gbe(src_ip_base+'%d'%(100+xeng.band), dest_ip, dest_mac, port)

    def all_fengs(self, method, *args, **kwargs):
        """
        Call FEngine method 'method' against all FEngine instances.
        Optional arguments are passed to the FEngine method calls.
        If an attribute rather than a method is specified, the attribute value
        will be returned as a list (one entry for each F-Engine). Otherwise the
        return value is that of the underlying FEngine method call.
        """
        self._logger.debug('Calling method %s against all F-engines in single-thread mode'%method)
        if callable(getattr(engines.FEngine,method)):
            return [getattr(feng, method)(*args, **kwargs) for feng in self.fengs]
        else:
            # no point in multithreading this
            return [getattr(feng,method) for feng in self.fengs]

    def all_fengs_multithread(self, method, *args, **kwargs):
        """
        Call FEngine method 'method' against all FEngine instances.
        Use a different thread per engine so calls don't block. This
        is useful for things like snapshot_get() where we might not
        want to wait for a snapshot to be grabbed before moving on to
        the next engine.
        Optional arguments are passed to the FEngine method calls.
        If an attribute rather than a method is specified, the attribute value
        will be returned as a list (one entry for each F-Engine). Otherwise the
        return value is that of the underlying FEngine method call.
        """
        self._logger.debug('Calling method %s against all F-engines in multi-thread mode'%method)
        return self.do_for_all(method, self.fengs, *args, **kwargs)

    def all_xengs(self, method, *args, **kwargs):
        """
        Call XEngine method 'method' against all XEngine instances.
        Optional arguments are passed to the XEngine method calls.
        If an attribute rather than a method is specified, the attribute value
        will be returned as a list (one entry for each X-Engine). Otherwise the
        return value is that of the underlying XEngine method call.
        """
        self._logger.debug('Calling method %s against all X-engines in single-thread mode'%method)
        if callable(getattr(engines.XEngine,method)):
            return [getattr(xeng,method)(*args, **kwargs) for xeng in self.xengs]
        else:
            return [getattr(xeng,method) for xeng in self.xengs]

    def all_fpgas(self, method, *args, **kwargs):
        """
        Call ROACH method 'method' against all ROACH instances.
        Optional arguments are passed to the ROACH  method calls.
        If an attribute rather than a method is specified, the attribute value
        will be returned as a list (one entry for each ROACH). Otherwise the
        return value is that of the underlying ROACH method call.
        ROACH instances subclass FpgaClient, so you can call any of their methods
        here.
        """
        self._logger.debug('Calling method %s against all ROACHes in single-thread mode'%method)
        if callable(getattr(roach.Roach,method)):
            return [getattr(fpga, method)(*args, **kwargs) for fpga in self.fpgas.values()]
        else:
            return [getattr(fpga,method) for fpga in self.fpgas.values()]

    def do_for_all(self, method, instances, *args, **kwargs):
        """
        Multithread calls of <method> against all instances <instances>.
        method: string name of method
        instances: list of instances to call method against.
        Optional args and kwargs are passed down to the methods.
        The return value is a list, as in [instance.method() for instance in instances]
        """
        self._logger.debug('Calling method %s in multi-thread mode'%method)
        t0 = time.time()
        if callable(getattr(instances[0],method)):
            q = Queue.Queue()
            for ii, inst in enumerate(instances):
                t = threading.Thread(target=_queue_instance_method, args=(q, ii, inst, method, args, kwargs))
                t.daemon = True
                t.start()
            t1 = time.time()
            rv = [None for inst in instances]
            for inst_n, inst in enumerate(instances):
                num, result = q.get()
                q.task_done()
                rv[num] = result
            t2 = time.time()
            q.join()
            t3=time.time()
            print 'thread starting: %.3fs; q getting %.3fs; joining %.3fs'%(t1-t0, t2-t1, t3-t2)
            return rv
        else:
            # no point in multithreading this
            return [getattr(inst, method) for inst in instances]
        pass

    def program_all(self):
        """
        Program all ROACHs. Since F and X engines share boards, programming via this method
        is preferable to programming via the F/XEngine instances.
        By default, F/X instances are rebuild after reprogramming, and FEngine ADCs are recalibrated.
        If reinitialise=False, no engines are instantiated (you will have to call initialise_f/x_engines
        manually.)
        """
        #for roach in self.fpgas.values():
        #    self._logger.info("Programming ROACH %s with boffile %s"%(roach.host,roach.boffile))
        #    roach.safe_prog()
        #    #self._logger.warning('SKIPPING QDR CALIBRATION -- TESTING ONLY!')
        #    #roach.calibrate_all_qdr()

        self._logger.info("Programming all ROACHs!")
        self.do_for_all('safe_prog', self.fpgas.values())
        self.do_for_all('program_all_sfp_phys', self.fpgas.values())
        self.do_for_all('calibrate_all_qdr', self.fpgas.values())

        # reprogramming messes with ctrl_sw, etc, so clean out the engine lists
        self._logger.info("Re-initializing engines")
        self.initialise_f_engines(passive=False)
        self.initialise_x_engines(passive=False)
#        self.set_chan_dests()

        self.all_fengs('calibrate_adc')
        #tock = time.time()
        #print 'Calibration time:', tock-tick
        #for fn,feng in enumerate(self.fengs):
        #     self._logger.warning('SKIPPING ADC CALIBRATION -- TESTING ONLY!')
        ##    feng.calibrate_adc()
    def get_array_config(self):
        pass
        


class spriteSbl(spriteDBE):
    """
    A subclass of spriteDBE for the single-ROACH, single-baseline correlator
    """

    def snap_corr_wide(self,wait=True,combine_complex=False):
        """
        Snap new correlations from bram.
        If wait is True, this method will return the correlations when they are ready,
        and block until this happens. Otherwise, the current values will be returned,
        whether or not this is a new accumulation.
        If combine_complex is true, complex data is returned as a complex number array.
        Otherwise, complex data is returned as a real array who's even and odd elements
        represent the imaginary and real correlation parts.
        The current accumulation number (MCNT) is read before and after reading data,
        if this changes during the read, nothing is returned.
        If the read is successful, the returned data is a dictionary with keys:
            corr00
            corr11
            corr01
            timestamp
        """
        if str(self.output_format) == 'q':
            array_fmt = np.int64
        elif str(self.output_format) == 'l':
            array_fmt = np.int32

        xeng0 = self.xengs[0]

        # flush vaccs
        xeng0.write_int('corr00_ctrl',0)
        xeng0.write_int('corr11_ctrl',0)
        xeng0.write_int('corr01_r_ctrl',0)
        xeng0.write_int('corr01_i_ctrl',0)
        time.sleep(0.01)
        xeng0.write_int('corr00_ctrl',1)
        xeng0.write_int('corr11_ctrl',1)
        xeng0.write_int('corr01_r_ctrl',1)
        xeng0.write_int('corr01_i_ctrl',1)
        
        if wait:
            mcnt = xeng0.read_int('mcnt_lsb')
            #sleep until there's a new correlation
            while xeng0.read_int('mcnt_lsb') == mcnt:
                time.sleep(0.01)

        snap00 = np.zeros(self.n_chans*self.n_bands,dtype=array_fmt)
        snap11 = np.zeros(self.n_chans*self.n_bands,dtype=array_fmt)
        snap01 = np.zeros(2*self.n_chans*self.n_bands,dtype=array_fmt)

        for xn,xeng in enumerate(self.xengs):
            mcnt_msb = xeng.read_uint('mcnt_msb')
            mcnt_lsb = xeng.read_uint('mcnt_lsb')
            mcnt = (mcnt_msb << 32) + mcnt_lsb
            #pack_format = '>%d%s'%(self.n_chans,str(self.output_format))
            pack_format = '>%dq'%(self.n_chans) 
            c_pack_format = '>%d%s'%(2*self.n_chans,str(self.output_format))
            n_bytes = struct.calcsize(pack_format)
            
            snap00[0:self.n_chans]   = np.array(struct.unpack(pack_format,xeng.read('corr00_bram',n_bytes)))
            snap11[0:self.n_chans]   = np.array(struct.unpack(pack_format,xeng.read('corr11_bram',n_bytes)))
            snap01[0:self.n_chans]   = np.array(struct.unpack(pack_format,xeng.read('corr01_r_bram',n_bytes)))
            snap01[self.n_chans:2*self.n_chans]   = np.array(struct.unpack(pack_format,xeng.read('corr01_i_bram',n_bytes)))            
            

            
            if mcnt_lsb != xeng.read_uint('mcnt_lsb'):
                print mcnt_lsb, xeng.read_uint('mcnt_lsb')
                print "SNAP CORR: mcnt changed before snap completed!"
                return None

        return {'corr00':snap00,'corr11':snap11,'corr01':snap01,'timestamp':self.mcnt2time(mcnt)}

    

    def mcnt2time(self, mcnt):
        """
        Convert an mcnt to a UTC time based on the instance's sync_time attribute.
        """
        conv_factor = 16./(self.adc_clk)
        offset = 0 #formerly the sync time
        return offset + mcnt*conv_factor
