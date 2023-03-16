from radiomics import featureextractor, getTestCase
import os
import pandas as pd
import numpy as np

def dict2pd(lists):
    print(len(list(lists[0].keys())))
    df = pd.DataFrame(columns=list(lists[0].keys()), dtype=object)
    for i in lists:
        df = df.append(i, ignore_index=True)
    return df


def ExtractorRadiomics(path, ptList, pathSave, params, name='wt2.nii', mask='RSEG.nii'):
    Radiomics = []
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    for ptid in ptList:
        pathMo = os.path.join(path, ptid, name)
        pathMASK = os.path.join(path, ptid, mask)
        result = extractor.execute(pathMo, pathMASK)
        Radiomics.append(result)
    RadiomicsCSV = dict2pd(Radiomics)
    RadiomicsCSV.to_csv(pathSave)


if __name__ == '__main__':
    from options import FeatureExtractionOptions
    options = FeatureExtractionOptions()
    opts = options.parse()
    path = opts.data_root
    ptList = opts.ptid_list
    params = opts.params
    masks = opts.masks
    mos = opts.modalities
    dirs = opts.res
    for mo in mos:
        for mask in masks:
            newname = mo.replace('.nii', '') + '_' + mask.replace('.nii', '') + '.csv'
            pathSave = os.path.join(dirs, newname)
            ExtractorRadiomics(path, ptList, pathSave, params, name=mo, mask=mask)