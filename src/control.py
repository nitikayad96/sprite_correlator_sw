import struct
import socket
import os
import string
import config_redis
import yaml
import numpy as np

class AmiControlInterface(object):
    """
    A class for the interface between the AMI digital correlator
    and the original analogue correlator control machine.
    This handles passing meta data messages to the digital correlator
    and digital correlator data sets to the original pipeline
    """
    def __init__(self,config_file=None, rain_gauge=False):
        """
        Initialise the interface, based on the config_file provided, or the AMI_DC_CONF
        environment variable is config_file=None
        """
        self.redis_host = config_redis.JsonRedis('ami_redis_host')
        if config_file is None:
            self.config = yaml.load(self.redis_host.hget('config', 'conf'))
            host = self.redis_host.hget('config', 'host')
            fn = self.redis_host.hget('config', 'file')
            config_file = '%s:%s'%(host, fn)
        else:
            self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.parse_config_file()
        self.rain_gauge = rain_gauge
        self.meta_data = get_meta_struct(maxant=self.n_ants, )#maxagc=self.n_agcs)
        print 'meta data size:', self.meta_data.size
        self.data = DataStruct(n_chans=self.n_chans*self.n_bands,n_bls=self.n_bls, n_ants=self.n_ants, rain_gauge=self.rain_gauge)

    def __del__(self):
        try:
            self.close_sockets()
        except:
            pass
    def parse_config_file(self):
        """
        Parse the config file, saving some values as attributes for easy access
        """
        #relevant parameters
        self.control_ip = self.config['Configuration']['control_interface']['host']
        self.data_port  = self.config['Configuration']['control_interface']['data_port']
        self.meta_port  = self.config['Configuration']['control_interface']['meta_port']
        self.n_ants      = self.config['Configuration']['correlator']['hardcoded']['n_ants']
        #self.n_agcs      = self.config['Configuration']['control_interface']['n_agcs']
        self.n_chans     = self.config['Configuration']['correlator']['hardcoded']['n_chans']
        self.n_bands     = self.config['Configuration']['correlator']['hardcoded']['n_bands']
        self.n_bls       = (self.n_ants * (self.n_ants + 1)) / 2
    def _bind_sockets(self):
        """
        Bind the sockets to the data and metadata server
        """
        self.rsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def close_sockets(self):
        """
        close the sockets
        """
        self.tsock.close()
        self.rsock.close()
    def connect_sockets(self, timeout=30):
        """
        Connect the tx/rx sockets to the correlator control server
        """
        self._bind_sockets()
        self.rsock.settimeout(timeout)
        self.rsock.connect((self.control_ip,self.meta_port))
        self.rsock.settimeout(0.01)
        self.tsock.settimeout(timeout)
        self.tsock.connect((self.control_ip,self.data_port))
        self.tsock.settimeout(0.01)
    def try_recv(self):
        """
        Try and receive meta-data from the control server.
        Return None if the read times out, or 0 if the read
        is successful. Unpack read data into meta data attributes
        """
        try:
            d = self.rsock.recv(self.meta_data.size)
        except socket.timeout:
            return None
        if len(d) == self.meta_data.size:
            self.meta_data.extract_attr(d)
            return 0
    def try_send(self, timestamp, status, nsamp, d, rain_gauge=None):
        """
        Try and send a data set to the control server.
        Return 0 if successful, -1 if not (and close tx socket)
        """
        if rain_gauge is not None:
            data_str = self.data.pack(timestamp, status, nsamp, *np.append(d, rain_gauge))
        else:
            data_str = self.data.pack(timestamp, status, nsamp, *d)
        try:
            self.tsock.send(data_str)
            return 0
        except socket.error:
            print "lost TX connection"
            self.tsock.close()
            return -1

class Unpackable(object):
    def __init__(self, varname, fmt):
        """
        A simple class to hold named values. It also stores
        their format, to facilitate unpacking.
        varname: The name of the variable
        fmt: The format of the value, in python struct style (e.g. '>32L')
             This format will be broken into 'end' and 'fmt' attributes.
        """
        self.varname = varname
        if fmt[0] in ['>', '<', '!', '=', '@']:
            self.fmt = fmt[1:]
            self.end = fmt[0]
        else:
            self.fmt = fmt
            self.end = ''
        self.size = struct.calcsize(self.fmt)
        self.offset = 0

class UnpackableStruct(Unpackable):
    def __init__(self, varname, entries, end='!'):
        """
        A class to facilitate unpacking binary data.
        entries: A list of entries in the struct. These can either
                 be instances of the Unpackable class,
                 or instances of the UnpackableStruct class
        end: endianess '!', '>', '<', '=' or '@'. See python struct docs.
             This is the endianness with which values in the struct will be
             unpacked. In theory, nested UnpackableStruct instances may
             have different endianess, but I don't know why you would ever
             do this.
        """
        
        self.fmt = self._expand_fmt(entries)
        self.end = end
        Unpackable.__init__(self, varname, self.end + self.fmt)
        self.entries = entries
        self._gen_offsets()
        print 'Building struct %s with size %d bytes'%(varname, self.size)

        # allow access to entries in the struct directly by name
        for entry in self.entries:
            if hasattr(self, entry.varname):
                raise ValueError("Structure %s already has attribute '%s'!"%(self.varname, entry.varname))
            self.__setattr__(entry.varname, entry)

    def _expand_fmt(self, entries):
        """
        Generate the complete struct format string
        """
        fmt = ''
        for entry in entries:
            fmt += entry.fmt
        return fmt

    def _gen_offsets(self):
        """
        Generate the offsets of each entry in the struct,
        to allow unpacking later.
        """
        offset = 0
        for entry in self.entries:
            entry.offset = offset
            offset += entry.size

    def extract_attr(self, data, offset=0):
        """
        Recursively update the values held by the entries in the struct
        """
        #print self.varname, 'Extracting a total of %d bytes (%s) (data size = %d bytes)'%(self.size, self.fmt, len(data))
        self.dict_repr = {}
        for entry in self.entries:
            #print 'extracting:', entry.varname, 'offset:', entry.offset, 'size', entry.size
            if isinstance(entry, UnpackableStruct):
                #print entry.varname, 'is a struct -- recursing'
                self.dict_repr[entry.varname] = entry.extract_attr(data, offset=offset+entry.offset)
            else:
                #print 'Extracting', entry.fmt, 'struct offset', offset, 'entry offset', entry.offset
                val = struct.unpack_from((entry.end or self.end) + entry.fmt, data, offset + entry.offset)
                if entry.fmt.endswith('s'):
                    val = [str(v.split('\x00')[0] or 'XXX') for v in val] #first part of string upto null byte. Default to 'XXX' to save zero-length string headaches
                if len(val) == 1:
                    entry.val = val[0]
                else:
                    entry.val = val
                self.dict_repr[entry.varname] = entry.val
        return self.dict_repr

def get_meta_struct(maxant=10, maxsrc=16, maxagc=40):
   tel_def = [
       Unpackable('ax',    '!%dd'%maxant),
       Unpackable('ay',    '!%dd'%maxant),
       Unpackable('az',    '!%dd'%maxant),
       Unpackable('tsys',  '!%df'%maxant),
       Unpackable('rain',  '!%df'%maxant),
       Unpackable('ant',   '!%di'%maxant),
       Unpackable('freq',  '!f'),
       Unpackable('array', '!i'),
       Unpackable('nant',  '!i'),
       Unpackable('nbase', '!i'),
   ]

   tel_def_str = UnpackableStruct('tel_def', tel_def, end='!')

   obs_def = [
       Unpackable('name',     '!32s'),
       Unpackable('file',     '!64s'),
       Unpackable('observer', '!32s'),
       Unpackable('comment',  '!80s'),
       Unpackable('ut1utc',   '!d'),
       Unpackable('mode',     '!i'),
       Unpackable('nstep',    '!i'),
       Unpackable('nstepx',   '!i'),
       Unpackable('nstepy',   '!i'),
       Unpackable('stepx',    '!i'),
       Unpackable('stepy',    '!i'),
       Unpackable('intsam',   '!i'),
       Unpackable('dummy',    '!i'),
   ]

   obs_def_str = UnpackableStruct('obs_def', obs_def, end='!')

   src_def = [
       Unpackable('name',   '!' + '16s'*maxsrc),
       Unpackable('epoch',  '!%di'%maxsrc),
       Unpackable('raref',  '!%dd'%maxsrc),
       Unpackable('decref', '!%dd'%maxsrc),
       Unpackable('raobs',  '!%dd'%maxsrc),
       Unpackable('decobs', '!%dd'%maxsrc),
       Unpackable('flux',   '!%df'%maxsrc),
       Unpackable('nsrc',   '!i'),
       Unpackable('dummy',  '!i'),
   ]

   src_def_str = UnpackableStruct('src_def', src_def, end='!')
   
   dcor_out = [
       Unpackable('timestamp', '!i'),
       Unpackable('obs_status','!i'),
       Unpackable('nsamp',     '!i'),
       Unpackable('nsrc' ,     '!i'),
       Unpackable('noff' ,     '!i'),
       Unpackable('dummy',     '!i'),
       Unpackable('smp_last',  '!d'),
       Unpackable('smp_ra',    '!d'),
       Unpackable('smp_dec',   '!d'),
       Unpackable('ha_reqd',   '!%di'%maxant),
       Unpackable('ha_read',   '!%di'%maxant),
       Unpackable('dec_reqd',  '!%di'%maxant),
       Unpackable('dec_read',  '!%di'%maxant),
       Unpackable('tcryo',     '!%di'%maxant),
       Unpackable('pcryo',     '!%di'%maxant),
       Unpackable('agc',       '!%di'%maxant),
       Unpackable('delay',     '!%di'%maxant),
       tel_def_str,
       obs_def_str,
       src_def_str,
   ]
   
   dcor_out_str = UnpackableStruct('dcor_out', dcor_out, end='!')
   
   return dcor_out_str

        
class DataStruct(struct.Struct):
    """
    A subclass of Struct to encapsulate correlator data and timestamp
    """
    def __init__(self, n_chans=2048, n_bls=1, n_ants=10, rain_gauge=False):
        """
        Initialise a data structure for a timestamp, status flag, count number,
        and n_chans oof complex data.
        """
        if not rain_gauge:
            form = '!dii%dl'%(2*n_chans*n_bls)
        else:
            form = '!dii%dl%df'%((2*n_chans*n_bls), n_chans*n_ants)
        struct.Struct.__init__(self,form)
