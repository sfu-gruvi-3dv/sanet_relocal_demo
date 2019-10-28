import numpy as np
import os
import glob
import sys

base_dir = sys.argv[1] if len(sys.argv) > 1 else '/local-scratch/SUN3D'

# Read list
list_file = os.path.join(base_dir, 'SUN3Dv1.txt')
seq_lists = []
with open(list_file, 'r') as list_f:
    for seq_name in list_f:
        seq_name = seq_name.strip()
        seq_lists.append(seq_name)

for seq_name in seq_lists:
    img_dir = os.path.join(base_dir, seq_name, 'image')
    img_list = glob.glob(os.path.join(img_dir, '*.*'))
    for img_file in img_list:
        file_size = os.path.getsize(img_file)
        if file_size == 0:
            img_name = img_file.split('/')[-1].strip()
            print(seq_name + '/image/' + img_name)

    depth_dir = os.path.join(base_dir, seq_name, 'depth')
    depth_list = glob.glob(os.path.join(depth_dir, '*.png'))
    for depth_file in depth_list:
        file_size = os.path.getsize(depth_file)
        if file_size == 0:
            depth_name = depth_file.split('/')[-1].strip()
            print(seq_name + '/depth/' + depth_name)
