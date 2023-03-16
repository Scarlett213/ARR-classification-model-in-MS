from __future__ import absolute_import, division, print_function

import os
import argparse
class SegmentationOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="MS lesion segmentation options")
        self.parser.add_argument("--ptid_list",
                                 type=str,
                                 nargs='+',
                                 help="List containing patient folders with nii files")
        self.parser.add_argument('--data_root',
                                 type=str,
                                 help = 'Root directory of patient folder')
        self.parser.add_argument('--res_root',
                                 type= str,
                                 help = 'Root directory for storing all results')
        self.parser.add_argument('--best_model',
                                 type=str,
                                 help= 'npy file store Optimal model in three directions')
        self.parser.add_argument('--classdataroot',
                                 type=str,
                                 help= 'Data root directory of classification task')
        self.parser.add_argument('--batch_size',type= int, default=1)
        self.parser.add_argument('--epochs', type= int, default=100)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--min_delta', type=float, default=1e-5,
                                 help ='min_delta for ReduceLROnPlateau' )
        self.parser.add_argument('--min_lr', type=float, default=1e-6,
                                 help = 'min_lr for ReduceLROnPlateau')
        self.parser.add_argument('--pe', type=int, default=10,
                                 help = 'patience for EarlyStopping')
        self.parser.add_argument('--pr', type=int, default=5,
                                 help= 'patience for ReduceLROnPlateau')



    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
