from __future__ import absolute_import, division, print_function

import argparse
class FeatureExtractionOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="feature extraction options")
        self.parser.add_argument('--params',type=str,
                                 help='yaml files stored the feature extraction parameter')
        self.parser.add_argument("--ptid_list",
                                 type=str,
                                 nargs='+',
                                 help="List containing patient folders with nii files")
        self.parser.add_argument('--data_root',type=str,
                                 help="Folder for storing mask,MRI and PET data")
        self.parser.add_argument('--masks',type=str,nargs='+',
                                 help='nii format mask',
                                 default=['PREDMV.nii'])
        self.parser.add_argument('--modalities',type=str,nargs='+',
                                 help='nii format MRI and PET',
                                default=['wt2.nii', 'PET.nii'])
        self.parser.add_argument('--res',help='Folder to save extracted features')




    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
