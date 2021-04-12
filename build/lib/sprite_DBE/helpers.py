import numpy as np
import logging
import logging.handlers
import sys
import redis
import json
import socket
import struct

'''
A Redis-based log handler from:
http://charlesleifer.com/blog/using-redis-pub-sub-and-irc-for-error-logging-with-python/
'''
class RedisHandler(logging.Handler):
    def __init__(self, channel, conn, *args, **kwargs):
        logging.Handler.__init__(self, *args, **kwargs)
        self.channel = channel
        self.redis_conn = conn

    def emit(self, record):
        attributes = [
            'name', 'msg', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'process', 'processName',
        ]
        record_dict = dict((attr, getattr(record, attr)) for attr in attributes)
        record_dict['formatted'] = self.format(record)
        try:
            self.redis_conn.publish(self.channel, json.dumps(record_dict))
        except UnicodeDecodeError:
            self.redis_conn.publish(self.channel, 'UnicodeDecodeError on emit!')
        except redis.RedisError:
            pass
    
def uint2int(d,bits,bp,complex=False):
    """
    Convert unsigned integers to signed values and return them
    d: array of unsigned data
    bits: number of bits in output
    bp: binary point of output data
    complex: True if input data follows casper standard complex format.
    False if data should be interpreted as real
    """
    if complex:
        dout_r = (np.array(d) & (((2**bits)-1)<<bits)) >> bits
        dout_i = np.array(d) & ((2**bits)-1)
        dout_r = uint2int(dout_r,bits,bp,complex=False)
        dout_i = uint2int(dout_i,bits,bp,complex=False)
        return dout_r + 1j*dout_i
    else:
        dout = np.array(d,dtype=float)
        dout[dout>(2**(bits-1))] -= 2**bits
        dout /= 2**bp
        return dout

def dbs(x):
    """
    return 10*log10(x)
    """
    return 10*np.log10(x)

def slice(val,lsb,width=1):
    """
    Return bits lsb+width-1 downto lsb of val
    If the output width is 1 bit, convert result to bool.
    """
    out = (val & ((2**width - 1) << lsb)) >> lsb
    if width == 1:
        return bool(out)
    else:
        return out

def add_default_log_handlers(logger, redishostname='ami_redis_host', fglevel=logging.INFO, bglevel=logging.INFO):
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)20s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(fglevel)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
    syslog_handler.setLevel(bglevel)
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)

    redis_host = redis.Redis(redishostname, socket_timeout=1)
    try:
        redis_host.ping()
    except redis.ConnectionError:
        logger.warn("Couldn't connect to redis server at %s"%redishostname)
        return logger

    redis_handler = RedisHandler('log-channel', redis_host)
    redis_handler.setLevel(bglevel)
    redis_handler.setFormatter(formatter)
    logger.addHandler(redis_handler)

    logger.info("Logger %s created..."%logger.name)

    return logger

def get_unscramble_map(n=11, m=3):
   """
   Generate the unscramble map for an 2**n point FFT
   with 2**m simultaneous inputs
   """
   out = np.zeros(2**n, dtype=int)
   for i in range(2**m):
       out[i * 2**(n-m) : (i+1) * 2**(n-m)] = np.arange(i, 2**n, 2**m)
   return out

def get_scramble_map(n=11, m=3):
   """
   generate the scramble map for a 2**n point FFT
   with 2**m simultaneous input.
   The scramble map is such that (eg)
   scramble_map[j] returns the frequency bin
   which FPGA channel index j represents.
   It is naively calculated by calling get_unscramble_map
   and looping over the indices to reverse the mapping
   """
   descr_map = get_unscramble_map(n=n, m=m)
   out = np.zeros(2**n, dtype=int)
   for i in range(2**n):
       out[descr_map[i]] = i
   return out

def descramble_spectra(d, n=11, m=3):
   """
   descramble a spectra, d, with 2**n points
   obtained by doing an fft with 2**m
   simultaneous inputs
   """
   descr_map = get_unscramble_map(n=n, m=m)
   return d[descr_map]

def ip_str2int(ip):
    return struct.unpack("!L", socket.inet_aton(ip))[0]

def ip_int2str(ip):                                                               
    return socket.inet_ntoa(struct.pack("!I", ip)) 

