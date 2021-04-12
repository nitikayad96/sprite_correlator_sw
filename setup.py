from setuptools import setup

__version__ = '0.1'

setup(name = 'sprite_DBE',
      version = __version__,
      description = 'Control software / scripts for the roach SPRITE digital correlator',
      #url = "https://github.com/jack-h/ami_correlator_sw",
      requires    = ['numpy', 'corr'],
      provides    = ['sprite_DBE'],
      package_dir = {'sprite_DBE':'src'},
      packages    = ['sprite_DBE'],
)

