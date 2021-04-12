"""
Define a status dictionary here, which encodes the names of flags in the F-engine
status registers. Each entry should appear as a key to the status dictionary.
This key should hold a dictionary itself, with 'start_bit', 'width' and 'default'
keys.
start_bit : the first (lowest) bit of the flag in the F-engine status register
width : the width, in bits, of the flag
default: The value the flag is expected to hold if the F-engine is operating correctly
(The default value is used to highlight suspicious flag values to the user)

e.g.:
status = {'feng_ok'    : {'start_bit':0, 'width':1, 'default':True},
          'feng_broken': {'start_bit':1, 'width':1, 'default':False}}
"""

status = {
          'quant_or'          :{'start_bit':0,  'width':1, 'default':False},
          'fft_or'            :{'start_bit':1,  'width':1, 'default':False},
          'adc_or'            :{'start_bit':2,  'width':1, 'default':False},
          'ct_error'          :{'start_bit':3,  'width':1, 'default':False},
          'adc_disable'       :{'start_bit':4,  'width':1, 'default':False},
          'clk_bad'           :{'start_bit':5,  'width':1, 'default':False}, 
          'dram_bad'          :{'start_bit':6,  'width':1, 'default':False},
          'xaui_of'           :{'start_bit':7,  'width':1, 'default':False},
          'xaui_down'         :{'start_bit':8,  'width':1, 'default':False},
          'sync_val'          :{'start_bit':9,  'width':2, 'default':0    },
          'armed'             :{'start_bit':11, 'width':1, 'default':False},
          'phase_switch_on'   :{'start_bit':12, 'width':1, 'default':True },
          'noise_switch_on'   :{'start_bit':13, 'width':1, 'default':True },
          }
