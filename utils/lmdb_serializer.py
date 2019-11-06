# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-11-06 10:10:07
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-11-06 14:12:59
# @Email:  cshzxie@gmail.com

import numpy as np
import os
import pcl
import sys

from tqdm import tqdm
from tensorpack import dataflow


def main():
    if not len(sys.argv) == 3:
        print('Usage: python lmdb_serializer.py lmdb_file_path output_folder')
        sys.exit(1)

    lmdb_file_path = sys.argv[1]
    output_base_folder = sys.argv[2]

    df = dataflow.LMDBSerializer.load(lmdb_file_path, shuffle=False)
    df.reset_state()
    for d in tqdm(df, leave=False):
        ids = d[0].split('_')
        category_id = ids[0]
        model_id = ids[1]
        idx = len(ids) - 3

        partial_output_folder = os.path.join(output_base_folder, 'partial', category_id, model_id)
        complete_output_folder = os.path.join(output_base_folder, 'complete', category_id)
        if not os.path.exists(partial_output_folder):
            os.makedirs(partial_output_folder)
        if not os.path.exists(complete_output_folder):
            os.makedirs(complete_output_folder)

        p = pcl.PointCloud()
        p.from_array(d[1].astype(np.float32))
        p.to_file(os.path.join(partial_output_folder, '%02d.pcd' % idx).encode())
        p.from_array(d[2].astype(np.float32))
        p.to_file(os.path.join(complete_output_folder, '%s.pcd' % model_id).encode())


if __name__ == '__main__':
    main()