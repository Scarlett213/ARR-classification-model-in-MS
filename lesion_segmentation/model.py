import tensorflow.keras.backend as K
K.set_image_data_format("channels_last")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D,concatenate, UpSampling2D, Dropout


def MMFF(MR_FM, PET_FM, nc1, name):
    name = 'mmff' + str(name) + '_'
    MR_FM01 = Conv2D(nc1, 1, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c1')(MR_FM)
    PET_FM01 = Conv2D(nc1, 1, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c2')(
        PET_FM)
    MR_FM13 = Conv2D(nc1, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c3')(MR_FM01)
    PET_FM13 = Conv2D(nc1, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c4')(
        PET_FM01)
    fm = concatenate([MR_FM13, PET_FM13], axis=3)
    return fm


def MSFU(high, low, nc1, name):
    name = 'msfu' + str(name) + '_'
    low01 = Conv2D(nc1, 1, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c1')(low)
    low12 = Conv2D(nc1, 2, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c2')(
        UpSampling2D(size=(2, 2))(low01))
    merge = concatenate([high, low12], axis=3)
    merge = Conv2D(nc1, 1, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c3')(merge)
    merge = Conv2D(nc1, 3, activation='relu', padding='same', kernel_initializer='he_normal', name=name + 'c4')(merge)
    return merge

def creatcnn(input_size):
    MRinputs = Input(input_size, name='MRinputs')
    PETinputs = Input(input_size, name='PETinputs')
    #MR down sample
    conv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv11")(MRinputs)
    conv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv12")(conv11)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool1")(conv12)
    conv21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv21")(pool1)
    conv22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv22")(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool2")(conv22)
    conv31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv31")(pool2)
    conv32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv32")(conv31)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool3")(conv32)
    conv41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv41")(pool3)
    conv42 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv42")(conv41)
    drop4 = Dropout(0.5, name="drop4")(conv42)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool4")(drop4)
    conv51 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv51")(pool4)
    conv52 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv52")(conv51)
    drop5 = Dropout(0.5, name="drop5")(conv52)

    # PET down sample
    PETconv11 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv11")(
        PETinputs)
    PETconv12 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv12")(
        PETconv11)
    PETpool1 = MaxPooling2D(pool_size=(2, 2), name="PETpool1")(PETconv12)
    PETconv21 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv21")(
        PETpool1)
    PETconv22 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv22")(
        PETconv21)
    PETpool2 = MaxPooling2D(pool_size=(2, 2), name="PETpool2")(PETconv22)
    PETconv31 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv31")(
        PETpool2)
    PETconv32 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv32")(
        PETconv31)
    PETpool3 = MaxPooling2D(pool_size=(2, 2), name="PETpool3")(PETconv32)
    PETconv41 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv41")(
        PETpool3)
    PETconv42 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv42")(
        PETconv41)
    PETdrop4 = Dropout(0.5, name="PETdrop4")(PETconv42)
    PETpool4 = MaxPooling2D(pool_size=(2, 2), name="PETpool4")(PETdrop4)
    PETconv51 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv51")(
        PETpool4)
    PETconv52 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="PETconv52")(
        PETconv51)
    PETdrop5 = Dropout(0.5, name="PETdrop5")(PETconv52)

    # MMFF
    fm1 = MMFF(conv12, PETconv12, 16 * 2, 1)
    fm2 = MMFF(conv22, PETconv22, 32 * 2, 2)
    fm3 = MMFF(conv32, PETconv32, 64 * 2, 3)
    fm4 = MMFF(drop4, PETdrop4, 128 * 2, 4)
    fm5 = MMFF(drop5, PETdrop5, 256 * 2, 5)
    # MSFU
    msfu1 = MSFU(fm4, fm5, 512, 1)
    msfu2 = MSFU(fm3, msfu1, 256, 2)
    msfu3 = MSFU(fm2, msfu2, 128, 3)
    msfu4 = MSFU(fm1, msfu3, 64, 4)
    msfu4 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='all1')(msfu4)
    msfu4 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal', name='all2')(msfu4)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='all3')(msfu4)
    conv10 = Conv2D(1, 1, activation='sigmoid', name='output')(conv9)
    model = Model(inputs=[MRinputs, PETinputs], outputs=conv10)
    return model