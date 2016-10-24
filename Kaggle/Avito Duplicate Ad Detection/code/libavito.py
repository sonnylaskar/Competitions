#### Copyright 2016 Mikel Bober-Irizar, Sonny Laskar & Peter Borrmann // TheQuants
#### Avito Duplicate Ad Detection
# Author: Mikel
# This file contains various functions which are used in multiple scripts

from imp import load_source
from time import time
import sys

# Terminal output colours for use in scripts
class c:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Function to read the config file
def read_config():
    conf = load_source('config.cfg', 'config.cfg')
    conf.nthreads = conf.model_nthreads
    conf.debug = 0
    # except Exception as e:
    #     #print(bcol.FAIL + 'Failed to parse config file:' + bcol.END)
    #     print(e.message, e.args)
    #     raise Exception(bcol.FAIL + 'Failed to parse config file:' + bcol.END)
    return conf

# Just an alias
def get_config():
    return read_config()

# Function which reads '--train' or '--test' launch arguments
def get_mode(argv, name='Script'):
    if len(argv) != 2:
        raise RuntimeError(name + ' must be called with either --train or --test')
    if argv[1] == '--train':
        mode = 0
    elif argv[1] == '--test':
        mode = 1
    else:
        raise RuntimeError(name + ' must be called with either --train or --test')
    assert mode == 0 or mode == 1
    return mode

# Function which prints current status and time remaining:
def print_progress(k, start, o):
    if k != 0:
        dur_per_k = (time() - start) / k
        rem_dur = dur_per_k * (o - k)
        rem_mins = int(rem_dur / 60)
        rem_secs = rem_dur % 60
        toprint = str(k) + " items processed - " + str(rem_mins) + "m" + str(int(rem_secs)) + "s remaining.  "
        sys.stdout.write(toprint + '\r')
        sys.stdout.flush()

def print_elapsed(start):
    print(str(round(time() - start, 1)) + 's elapsed')
