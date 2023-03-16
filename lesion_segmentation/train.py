import tensorflow as tf
import cv2 as cv
from sklearn.model_selection import KFold
from train_utils import *
import os
kf = KFold(n_splits=5,shuffle=True)

from options import SegmentationOptions
options = SegmentationOptions()
opts = options.parse()

###################################################
# Record the size of the input picture in three directions
# The first one represents the size of the original 2D slice
# the others represent the size of the input to network after padding and cropping
###################################################

size={'X':[(145,121,1),(144,128,1),(144,128)],
      'Y':[(121,121,1),(128,128,1),(128,128)],
      'Z':[(121,145,1),(128,144,1),(128,144)]}


X = opts.ptid_list
dataroot = opts.data_root
resroot = opts.res_root
batch_size=opts.batch_size
lr=opts.lr
epochs=opts.epochs
min_delta=opts.min_delta
min_lr=opts.min_lr
pe=opts.pe
pr=opts.pr

for direction,input_size in size.items():
    for trab_index,test_index in kf.split(X):
        #Define variables
        pngroot=os.path.join(resroot,direction,'pngPlot')
        predMaskRoot=os.path.join(resroot,direction,'predMask')
        hdfroot=os.path.join(resroot,direction,'hdf5')
        npyroot=os.path.join(resroot,direction,'resultNpy')
        train=pick(trab_index,X)
        test=pick(test_index,X)
        title=index_title(test)
        path_train=os.path.join(dataroot,direction,'train')
        path_test=os.path.join(dataroot,direction,'test')
        hdfname=os.path.join(hdfroot,title)+'.hdf5'
        pngname=os.path.join(pngroot,title)
        npyname=os.path.join(npyroot,title)+'.npy'
        path_save=os.path.join(predMaskRoot,title)
        checkDir(path_save)
        #move dataset for 5-fold validaiton
        movepng_test2train(path_train,path_test)
        movepng_train2test(path_train,path_test,test)
        train_amount=len(os.listdir(os.path.join(path_train,'MR')))
        steps_per_epoch=np.round(train_amount*0.8/batch_size)
        validation_steps=np.round(train_amount*0.2/batch_size)
        test_amount=len(os.listdir(os.path.join(path_test,'MR')))
        #training and test
        image_generator_train,image_generator_val,PETimage_generator_train,PETimage_generator_val,mask_generator_train,mask_generator_val=GEN2D(path_train,batch_size,input_size[2])
        result=MyTrainAndTest(lr,input_size[1],steps_per_epoch,validation_steps,epochs,
                              title,hdfname,pngname,path_test,test_amount,min_delta, min_lr,pe,pr,
                              image_generator_train,image_generator_val,PETimage_generator_train,
                              PETimage_generator_val,mask_generator_train,mask_generator_val)
        #recording test result and visualize predicted 2D slice
        np.save(npyname,result)
        saveResult(path_save, os.path.join(path_test, 'MR'), result)
        movepng_test2train(path_train, path_test)
