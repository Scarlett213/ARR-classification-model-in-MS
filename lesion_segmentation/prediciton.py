import numpy as np
import os
import tensorflow as tf
from options import SegmentationOptions
from model import creatcnn
from train_utils import saveResult, testGenerator
from evaluation_util import reconstructionPRE, binarys
from prediction_utils import *
import pandas as pd
options = SegmentationOptions()
opts = options.parse()
best_model=np.load(opts.best_model,allow_pickle=True).item()
classdataroot = opts.classdataroot
resroot = opts.res_root
ptids = opts.ptid_list
###################################################
# Record the size of the input picture in three directions
# The first one represents the size of the original 2D slice
# the second represents the size of the input to network after padding and cropping
# the third one represents the reconstruction axis
###################################################
size={'X':[(145,121,1),(144,128),2],
      'Y':[(121,121,1),(128,128),1],
      'Z':[(121,145,1),(128,144),0]}

#Load the optimal model for 2D slice prediction
for direction,v in size.items():
    input_size=v[1]+(1,)
    model_weight=best_model[direction]+'.hdf5'
    weight_dir=os.path.join(resroot,direction,'hdf5',model_weight)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            sess.run(tf.global_variables_initializer())  # tf.initialize_all_variables())
            sess.run(tf.local_variables_initializer())
            model=creatcnn(input_size)
            model.load_weights(weight_dir, by_name=False)
            path_save=os.path.join(classdataroot,direction,'PRED')
            path_test=os.path.join(classdataroot,direction,'test')
            testGen=testGenerator(path_test)
            testSize=len(os.listdir(os.path.join(path_test,'MR')))
            result=model.predict_generator(testGen,testSize,verbose=1)
            saveResult(path_save,os.path.join(path_test,'MASK'),result)


#3d reconstruction
df=pd.DataFrame(columns=['X','Y','Z'],index=ptids, dtype=object)
df1=pd.DataFrame(columns=['X','Y','Z'],index=ptids,dtype=object)
df2=pd.DataFrame(columns=['X','Y','Z'],index=ptids,dtype=object)
for direction,v in size.items():
    path_pred=os.path.join(classdataroot,direction,'PRED')
    sizes=v[1]
    axis=v[2]
    for ptid in ptids:
        sliceList,prd, SliceNum, slice2Dindex=reconstructionPRE(ptid,path_pred,sizes,axis)
        df.loc[ptid,direction]=prd # save 2D pred res
        df1.loc[ptid,direction]=slice2Dindex #保存有mask的切片序号
# padding to the same size of ground truth
kk=['X','Y','Z']
for k,v in df.iterrows():
    for i in range(3):
        p=transs(v[i],size[kk[i]][2])
        num=df1.loc[k,kk[i]]
        p=add0(p,size[kk[i]][2],num)
        df2.loc[k,kk[i]]=p
#major voting
for k,v in df2.iterrows():
    path=os.path.join(classdataroot,k,'PREDMV.nii')
    path_gt = os.path.join(classdataroot ,k, 'SEG.nii')
    x=v[0]
    y=v[1]
    z=v[2]
    mv=MV(x,y,z)
    _, aff = loadNii(path_gt)
    save_nii(mv, path, aff)