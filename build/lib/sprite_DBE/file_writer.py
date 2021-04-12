import os
import h5py
import yaml
import config_redis
import numpy as np

type_unicode = h5py.special_dtype(vlen=unicode)

class H5Writer(object):
    """
    A class to control writing data sets to hdf5 files.
    """
    def __init__(self,config_file=None,band='low'):
        """
        Instatiate a writer object, based on the provided config file, or the
        AMI_DC_CONF variable if none is provided.
        band: 'high' or 'low'. The sideband of the data being written. 
        """
        self.redis_host = config_redis.JsonRedis('ami_redis_host')
        if config_file is None:
            self.config = yaml.load(self.redis_host.hget('config', 'conf'))
        else:
            self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        self.band = band
        self.parse_config_file()
        self.datasets = {}
        self.datasets_index = {}
        self.fh=None
    def parse_config_file(self):
        """
        Parse the config file, saving some values to attributes for easy access
        """
        #some common params
        self.n_ants  = self.config['Configuration']['correlator']['hardcoded']['n_ants']
        self.n_bands = self.config['Configuration']['correlator']['hardcoded']['n_bands']
        self.n_inputs= self.config['Configuration']['correlator']['hardcoded']['inputs_per_board']
        self.n_chans = self.config['Configuration']['correlator']['hardcoded']['n_chans']
        self.n_pols  = self.config['Configuration']['correlator']['hardcoded']['n_pols']
        self.output_format  = self.config['Configuration']['correlator']['hardcoded']['output_format']
        self.acc_len  = self.config['XEngine']['acc_len']
        self.data_path  = self.config['Configuration']['correlator']['runtime']['data_path']
        self.adc_clk  = self.config['FEngine']['adc_clk']
        self.lo_freq  = self.config['FEngine']['mix_freq']
        self.n_bls = (self.n_ants * (self.n_ants + 1))/2

        self.roaches = set([node['host'] for node in self.config['FEngine']['nodes']+self.config['XEngine']['nodes']])
        if self.n_bands == 2:
            self.center_freq = self.lo_freq
        elif self.bands[0] == 'low':
            self.center_freq = self.lo_freq - self.adc_clk/4.
        elif self.bands[1] == 'high':
            self.center_freq = self.lo_freq + self.adc_clk/4.
        self.bandwidth = self.adc_clk/2. * self.n_bands
        #shortcuts to sections
        self.c_testing = self.config['Configuration']['correlator']['runtime']['testing']
        self.c_correlator = self.config['Configuration']['correlator']['runtime']
        self.c_correlator_hard = self.config['Configuration']['correlator']['hardcoded']
        self.c_global = self.config['Configuration']
        # geometry
        self.ant_locs = [[0., 0., 0.,] for ant in range(self.n_ants)]
        for i in range(self.n_ants):
            for ant in self.config['Antennas']:
                if ant['index'] == i:
                    self.ant_locs[i] = ant['loc']
        self.array_loc = [self.config['Array']['lat'], self.config['Array']['lon']]

    def start_new_file(self,name):
        """
        Close the current file if necessary, and start a new one with the provided name.
        """
        # close old file if necessary
        if self.fh is not None:
            self.close_file()
        self.fh = h5py.File(self.data_path+'/'+name,'w')
        self.write_fixed_attributes()
        self.datasets = {}
        self.datasets_index = {}
    def set_bl_order(self,order):
        """
        Set the baseline order, which is written to the hdf5 file.
        """
        self.bl_order = order
    def write_fixed_attributes(self):
        """
        Write static meta-data to the current h5 file.
        This data is:
            n_chans
            n_pols
            n_bls
            n_ants
            bl_order
            center_freq
            bandwidth
            array_loc
            antenna_locations
        """
        self.fh.attrs['n_chans'] = self.n_chans*self.n_bands
        self.fh.attrs['n_pols'] = self.n_pols
        self.fh.attrs['n_bls'] = self.n_bls
        self.fh.attrs['n_ants'] = self.n_ants
        self.fh.create_dataset('bl_order',shape=[self.n_bls,2],dtype=int,data=self.bl_order)
        self.fh.attrs['center_freq'] = self.center_freq
        self.fh.attrs['bandwidth'] = self.bandwidth
        self.fh.attrs['array_loc'] = self.array_loc
        self.fh.attrs['ant_locs'] = self.ant_locs

    def add_new_dataset(self,name,shape,dtype):
        """
        Add a new data set to the current h5 file.
        name: name of dataset
        shape: shape of data set ([dim0,dim1,...,dimN])
        dtype: data type of dataset.
        """
        self.fh.create_dataset(name,[1] + ([] if list(shape) == [1] else list(shape)), maxshape=[None] + ([] if list(shape) == [1] else list(shape)),dtype=dtype)
        self.datasets[name] = name
        self.datasets_index[name] = 0

    def append_data(self,name,shape,data,dtype):
        """
        Add data to the h5 file, starting a new data set or appending
        to an existing one as required.
        name: name of dataset to append to / create
        shape: shape of data
        data: data values to be written
        dtype: data type
        """
        if dtype is unicode:
            dtype = type_unicode
        if name not in self.datasets.keys():
            self.add_new_dataset(name,shape,dtype)
        else:
            self.fh[name].resize(self.datasets_index[name]+1,axis=0)
        self.fh[name][self.datasets_index[name]] = data
        self.datasets_index[name] += 1

    def close_file(self):
        """
        Close the currently open h5 file
        """
        if self.fh is not None:
            self.fh.close()
        self.fh = None

    def add_attr(self,name,val,shape=None, dtype=None):
        """
        Add an attribute with the supplied name and value to the current h5 file
        """
        try:
            if val.dtype.type is np.unicode_:
                dtype = type_unicode
        except AttributeError:
            pass
        try:
            if type(val[0]) is unicode:
                dtype = type_unicode
        except TypeError:
            pass

        if type(val) is unicode:
           dtype = type_unicode
        self.fh.attrs.create(name, val, shape=shape, dtype=dtype)
