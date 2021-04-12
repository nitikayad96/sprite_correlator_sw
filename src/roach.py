import struct, logging, time
import corr.katcp_wrapper as katcp
import helpers
import numpy as np
import re
import qdr

logger = helpers.add_default_log_handlers(logging.getLogger(__name__))

class Roach(katcp.FpgaClient):
    '''
    A minor expansion on the FpgaClient class adds a few methods.
    '''
    def __init__(self, roachhost, port=7147, boffile=None, logger=logger):

        self._logger = logger.getChild('(%s)'%roachhost)
        self._logger.handlers = logger.handlers
                
        katcp.FpgaClient.__init__(self,roachhost, port, logger=self._logger)
        self.boffile = boffile
        # self._logger should be set by the FpgaClient class,
        # but looking at the katcp code I'm not convinced
        # that it is properly passed up the chain of classes
        # TODO: is there a typo in the katcp CallbackClient class __init__?
        # Should the superclass be constructed with logger=log, not logger=logger?

    def snap(self,name,format='L',**kwargs):
        """
        A wrapper for snapshot_get(name, **kwargs), which decodes data into a numpy array, based on the format argument.
        Big endianness is assumped, so only pass the format character. (i.e., 'L' for unsigned 32 bit, etc).
        See the python struct manual for details of available formats.
        """

        self._logger.debug('Snapping register %s (assuming format %s)'%(name, format))
        n_bytes = struct.calcsize('=%s'%format)
        #d = self.snapshot_get(name, **kwargs)
        d = self.snapshot_get(name, man_trig=True)
        self._logger.debug('Got %d bytes'%d['length'])
        return np.array(struct.unpack('>%d%s'%(d['length']/n_bytes,format),d['data']))

    def calibrate_all_qdr(self, verbosity=1):
        """
        search for qdr's in the device list of the currently running firmware and
        attempt to calibrate them all
        """
        self._logger.info('Searching for QDRs to calibrate on roach %s'%self.host)
        dev_list = self.listdev()
        for dev in dev_list:
            if re.search('qdr[0-9]_memory', dev) is not None:
                self._logger.info('Found QDR: %s'%dev)
                self.calibrate_qdr(dev.rstrip('_memory'), verbosity=verbosity)
    
    def program_all_sfp_phys(self):
        """
        Try to program all SFP phys. Give up on any failure (probably not the desired behaviour,
        as failure could be due to no SFP present (ok), or no phy binary (not ok).
        Even goes so far as to disable the FpgaClient logger so as to hide false errors.
        TODO: make smarter.
        """
        timeout = 30
        orig_log_level = self._logger.getEffectiveLevel()
        for mezz in range(2):
            for phy in range(2):
                # only allow critical messages, unless the phyprog call passes
                # in which case temporarily allow messages to pass an INFO
                self._logger.setLevel(logging.CRITICAL)
                try:
                    self._request('phyprog', timeout, '%d'%mezz, '%d'%phy)
                except RuntimeError:
                    continue
                self._logger.setLevel(orig_log_level)
                self._logger.info('Programmed SFP mezzanine %d, phy %d'%(mezz, phy))
        self._logger.setLevel(orig_log_level)

    def calibrate_qdr(self, qdrname, verbosity=2):
        self._logger.info('Attempting to calibrate %s'%qdrname)
        qdr_obj = qdr.Qdr(self, qdrname)
        qdr_obj.qdr_cal2(fail_hard=True, verbosity=verbosity)

    def safe_prog(self, check_clock=True):
        """
        A wrapper for the FpgaClient progdev method.
        This method checks the target boffile is available before attempting to program, and clears
        the FPGA before programming. A test write to the sys_scratchpad register is performed after programming.
        If check_clock=True, the FPGA clock rate is estimated via katcp and returned in MHz.
        """
        self._logger.info("Programming ROACH %s with boffile %s"%(self.host, self.boffile))
        if self.boffile not in self.listbof():
            self._logger.critical("boffile %s not available on ROACH %s"%(self.boffile,self.host))
            raise RuntimeError("boffile %s not available on ROACH %s"%(self.boffile,self.host))
        #try:
        #    self.progdev('')
        #    time.sleep(0.1)
        #except:
        #    self._logger.warning("progdev('') call failed. Continuing anyway")
        self.progdev(self.boffile)
        time.sleep(0.1)
        # write_int automatically does a read check. The following call will fail
        # if the roach hasn't programmed properly
        self.write_int('sys_scratchpad',0xdeadbeef)
        if check_clock:
            clk = self.est_brd_clk()
            self._logger.info('Board clock is approx %.3f MHz'%clk)
            return clk
        else:
            return None

    def set_boffile(self,boffile):
        """
        Set the self.boffile attribute, which is used in safe_prog calls.
        """
        self.boffile=boffile
