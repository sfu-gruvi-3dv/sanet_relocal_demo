import numpy as np
import os
import glob
import sys

base_dir = sys.argv[1] if len(sys.argv) > 1 else '/mnt/Exp_2/SUN3D'

# Read list
list_file = os.path.join(base_dir, 'SUN3Dv1 (copy).txt')
seq_lists = []
with open(list_file, 'r') as list_f:
    for seq_name in list_f:
        seq_name = seq_name.strip()
        if os.path.exists(os.path.join(base_dir, seq_name)):
            seq_lists.append(seq_name)

# Write list
list_file = os.path.join(base_dir, 'SUN3Dv1_train.txt')
with open(list_file, 'w') as list_f:
    for i, seq_name in enumerate(seq_lists):
        if i > 0:
            list_f.write('\n')
        list_f.write(seq_name)
