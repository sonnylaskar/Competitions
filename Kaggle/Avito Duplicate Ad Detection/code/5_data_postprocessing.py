import pandas as pd
import numpy as np
import libavito as a
import feather as f
import time

cache_loc = a.get_config().cache_loc

start = time.time()
print('Transforming training data ... ', end='', flush=True)
df = f.read_dataframe(cache_loc + 'final_featureSet_train.fthr')
df.replace([np.nan, None], -1, inplace=True)
df.replace([np.inf, -np.inf], 9999.99, inplace=True)
f.write_dataframe(df, cache_loc + 'final_featureSet_train.fthr')
del df
a.print_elapsed(start)

start = time.time()
print('Transforming testing data ... ', end='', flush=True)
df = f.read_dataframe(cache_loc + 'final_featureSet_test.fthr')
df.replace([np.nan, None], -1, inplace=True)
df.replace([np.inf, -np.inf], 9999.99, inplace=True)
f.write_dataframe(df, cache_loc + 'final_featureSet_test.fthr')
a.print_elapsed(start)

