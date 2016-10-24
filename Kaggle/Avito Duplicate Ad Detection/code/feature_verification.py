import pandas as pd
import numpy as np
import libavito as a
from multiprocessing import Pool

df1 = pd.read_csv('')
df2 = pd.read_csv('')

def find_best_feature(c):
    ftr = df1[c].values

    high_correl = 0
    high_ftr = ''
    num_995 = 0
    for c2 in df2.columns:
        cor = np.corrcoef(ftr, df2[c2])[0, 1]
        if cor > 0.995:
            num_995 += 1
        if cor > high_correl:
            high_correl = cor
            high_ftr = c2

    return high_correl, high_ftr, num_995

for c in df1.columns:
    hc, hf, n995 = find_best_feature(c)

    if hc == 1:
        print(a.c.OKGREEN + (c + ' -> ' + hf).ljust(60) + ' | CORREL 1' + a.c.END)
    elif hc > 0.995:
        print(a.c.OKBLUE + (c + ' -> ' + hf).ljust(60) + ' | CORREL ' + str(hc) + a.c.END)
    elif hc > 0.95:
        print(a.c.WARNING + (c + ' -> ' + hf).ljust(60) + ' | CORREL ' + str(hc) + a.c.END)
    else:
        print(a.c.FAIL + (c + ' -> ???? ').ljust(60) + ' | ' + str(hc) + ' ' + hf)
