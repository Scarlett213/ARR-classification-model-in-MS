from __future__ import absolute_import, division, print_function

import argparse
class ClassificaitonOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="classification options")
        self.parser.add_argument('--csv_path', default=str,
                                 help='Csv file to save the features of the lesion area')
        self.parser.add_argument('--classifiers', default=str,
                                 choices=['SVM', 'RF', 'KNN', 'LR'])
        self.parser.add_argument('--ARR_path', default=str,
                                 help='Path to the excel of patients ARR')
        self.parser.add_argument('save_path', default=str,
                                 help='Txt file to save selected features')
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options