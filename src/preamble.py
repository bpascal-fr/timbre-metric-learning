import time
import os
import sys
sys.path.append("../include")

import numpy             as np
import matplotlib        as mpl
import matplotlib.pyplot as plt
import pandas            as pd

from   math              import sqrt
from   joblib            import Parallel, delayed
from   scipy.io          import savemat, loadmat
from   scipy.stats       import iqr
from   scipy.integrate   import simps
from   pearson           import logpearson, dlogpearson, logpearson_nan, dlogpearson_nan
from   load_dataset      import load_data
from   optim             import bfgs_log_kernel, bfgs_log_kernel_w1, bfgs_log_kernel_nan


plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,amsfonts}')
mpl.rcParams['font.family'] = 'roman'