import redis
import helpers
import socket
import time
import yaml
import json
import logging

logger = helpers.add_default_log_handlers(logging.getLogger(__name__))

def write_config_to_redis(configfile):
    with open(configfile, 'r') as fh:
        confstr = fh.read()

    conf = yaml.load(confstr)
    redis_host = conf['Configuration']['redis_host']
    redis_port = conf['Configuration']['redis_port']

    r = redis.Redis(redis_host, port=redis_port)
    r.hset('config', 'file', configfile)
    r.hset('config', 'host', socket.gethostname())
    r.hset('config', 'time', time.asctime(time.gmtime()))
    r.hset('config', 'unixtime', time.time())
    r.hset('config', 'conf', conf)
    r.hset('config', 'confstr', confstr)

class JsonRedis(redis.Redis):
    '''
    As redis.Redis, but read and write json strings. Also make all
    set calls write hashes with a timestamp
    '''

    def get(self, name):
        try:
            val, updated = map(json.loads, self.hmget(name, ['val', 'updated']))
            return val
        except:
            if not self.exists(name):
                logger.error('Couldn\'t find redis key %s'%name)
            else:
                if not self.hexists(name, 'val'):
                    logger.error('Couldn\'t find value for redis key %s'%name)
                if not self.hexists(name, 'updated'):
                    logger.warning('Couldn\'t find update time for redis key %s'%name)
                try:
                    return json.loads(redis.Redis.get(self, name))
                except:
                    logger.error('Tried to read redis key %s as value (not hash) and failed'%name)
                

    def get_update_time(self, name):
        return json.loads(self.hget(name, 'updated'))

    def get_age(self, name):
        update_time = self.get_update_time(name)
        return time.time() - update_time

    def set(self, name, value, force_timestamp=False, **kwargs):
        '''
        JSONify the input and send to redis.
        Write as a hash containing the
        update time, with keyname 'last_update_time'
        '''
        if not force_timestamp:
            update_time = json.dumps(time.time())
        else:
            update_time = json.dumps(force_timestamp)
        if ('nx' in kwargs.keys()) and (not self.exists(name)):
            self.hmset(name, {'val':json.dumps(value), 'updated':update_time})
        elif ('xx' in kwargs.keys()) and (self.exists(name)):
            self.hmset(name, {'val':json.dumps(value), 'updated':update_time})
        else:
            self.hmset(name, {'val':json.dumps(value), 'updated':update_time})
            
        if 'ex' in kwargs.keys():
            self.expire(name, kwargs['ex'])
        elif 'px' in kwargs.keys():
            self.expire(name, kwargs['px'] / 1e3)

