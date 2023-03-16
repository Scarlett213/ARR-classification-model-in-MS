from FeartureSelect import *
from func import *
import pandas as pd
from options import FeatureSelectionOptions

if __name__ =='__main__':
    options = FeatureSelectionOptions()
    opts = options.parse()

    path=opts.csv_path
    radiomics = pd.read_csv(path)
    method = opts.selectMethods
    path_ARR= opts.ARR_path
    pathtxt=opts.save_path

    y = pd.read_excel(path_ARR)
    featureSel=chooseFea(method, radiomics, y)
    featureSelCor = DeleteCors(radiomics, featureSel, y)
    SaveFea(featureSelCor, pathtxt)

