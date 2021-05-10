import os
import cv2
import math
import pandas as pd
import numpy as np
import random
from random import choice, sample
from collections import defaultdict
from keras.models import load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def bright(img):
    contrast = random.uniform(0.9,1.1)
    brightness = random.randint(-30,30)
    img = np.uint8(np.clip((contrast*img + brightness), 0, 255))
    return img

def gauss(img):
    gauss = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)
    return gauss

def gray(img):
    gy = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gy = cv2.merge([gy,gy,gy])
    return gy 

def hflip(img):
    img = cv2.flip(img, 1)
    return img

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std  = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def read_img_fn(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_FN,IMG_SIZE_FN))
    img = np.array(img).astype(np.float)
    return prewhiten(img)

def read_img_vgg(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_VGG,IMG_SIZE_VGG))
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def read_img_fn_train(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_FN,IMG_SIZE_FN))
    
    # if(random.uniform(0, 1) <= 0.1):
        # img = gray(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = bright(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = gauss(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = hflip(img) 
        
    img = np.array(img).astype(np.float)
    return prewhiten(img)

def read_img_vgg_train(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(IMG_SIZE_VGG,IMG_SIZE_VGG))
    
    # if(random.uniform(0, 1) <= 0.1):
        # img = gray(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = bright(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = gauss(img) 
    if(random.uniform(0, 1) <= 0.2):
        img = hflip(img) 
        
    img = np.array(img).astype(np.float)
    return preprocess_input(img, version=2)

def gen_train(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 3)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1     = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1_FN  = np.array([read_img_fn_train(x) for x in X1])
        X1_VGG = np.array([read_img_vgg_train(x) for x in X1])

        X2     = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2_FN  = np.array([read_img_fn_train(x) for x in X2])
        X2_VGG = np.array([read_img_vgg_train(x) for x in X2])

        yield [X1_FN, X2_FN, X1_VGG, X2_VGG], labels

def gen_val(list_tuples, person_to_images_map, batch_size=16):
    ppl = list(person_to_images_map.keys())
    while True:
        batch_tuples = sample(list_tuples, batch_size // 3)
        labels = [1] * len(batch_tuples)
        while len(batch_tuples) < batch_size:
            p1 = choice(ppl)
            p2 = choice(ppl)

            if p1 != p2 and (p1, p2) not in list_tuples and (p2, p1) not in list_tuples:
                batch_tuples.append((p1, p2))
                labels.append(0)

        for x in batch_tuples:
            if not len(person_to_images_map[x[0]]):
                print(x[0])

        X1     = [choice(person_to_images_map[x[0]]) for x in batch_tuples]
        X1_FN  = np.array([read_img_fn(x) for x in X1])
        X1_VGG = np.array([read_img_vgg(x) for x in X1])

        X2     = [choice(person_to_images_map[x[1]]) for x in batch_tuples]
        X2_FN  = np.array([read_img_fn(x) for x in X2])
        X2_VGG = np.array([read_img_vgg(x) for x in X2])

        yield [X1_FN, X2_FN, X1_VGG, X2_VGG], labels
        
def RFIW_Model():
    input_1 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_2 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_3 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))
    input_4 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))

    x1 = model_facenet(input_1)
    x2 = model_facenet(input_2)
    x3 = model_vgg(input_3)
    x4 = model_vgg(input_4)
    
    x1 = Reshape((1, 1 ,128))(x1)
    x2 = Reshape((1, 1 ,128))(x2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x1t = Lambda(lambda tensor : K.square(tensor))(x1)
    x2t = Lambda(lambda tensor : K.square(tensor))(x2)
    x3t = Lambda(lambda tensor : K.square(tensor))(x3)
    x4t = Lambda(lambda tensor : K.square(tensor))(x4)
    
    merged_add_fn   = Add()([x1, x2])
    merged_add_vgg  = Add()([x3, x4])
    merged_sub1_fn  = Subtract()([x1,x2])
    merged_sub1_vgg = Subtract()([x3,x4])
    merged_sub2_fn  = Subtract()([x2,x1])
    merged_sub2_vgg = Subtract()([x4,x3])
    merged_mul1_fn  = Multiply()([x1,x2])
    merged_mul1_vgg = Multiply()([x3,x4])
    merged_sq1_fn   = Add()([x1t,x2t])
    merged_sq1_vgg  = Add()([x3t,x4t])
    merged_sqrt_fn  = Lambda(lambda tensor  : signed_sqrt(tensor))(merged_mul1_fn)
    merged_sqrt_vgg = Lambda(lambda tensor  : signed_sqrt(tensor))(merged_mul1_vgg)
    
    merged_add_vgg  = Conv2D(128 , [1,1] )(merged_add_vgg)
    merged_sub1_vgg = Conv2D(128 , [1,1] )(merged_sub1_vgg)
    merged_sub2_vgg = Conv2D(128 , [1,1] )(merged_sub2_vgg)
    merged_mul1_vgg = Conv2D(128 , [1,1] )(merged_mul1_vgg)
    merged_sq1_vgg  = Conv2D(128 , [1,1] )(merged_sq1_vgg)
    merged_sqrt_vgg = Conv2D(128 , [1,1] )(merged_sqrt_vgg)
    
    merged = Concatenate(axis=-1)([Flatten()(merged_add_vgg), (merged_add_fn), Flatten()(merged_sub1_vgg), (merged_sub1_fn),
                                   Flatten()(merged_sub2_vgg), (merged_sub2_fn), Flatten()(merged_mul1_vgg), (merged_mul1_fn), 
                                   Flatten()(merged_sq1_vgg), (merged_sq1_fn), Flatten()(merged_sqrt_vgg), (merged_sqrt_fn)])
    
    merged = Dense(100, activation="relu")(merged)  ## use_bias=False
    merged = Dropout(0.1)(merged)
    merged = Dense(25, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model([input_1, input_2, input_3, input_4], out)

    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.0001))
#    model.compile(loss=[focal_loss(alpha=.25, gamma=2)], metrics=['acc'], optimizer=Adam(0.0001))
#    model.summary()
    return model

def RFIW_Model2():
    input_1 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_2 = Input(shape=(IMG_SIZE_FN, IMG_SIZE_FN, 3))
    input_3 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))
    input_4 = Input(shape=(IMG_SIZE_VGG, IMG_SIZE_VGG, 3))

    x1 = model_facenet(input_1)
    x2 = model_facenet(input_2)
    x3 = model_vgg(input_3)
    x4 = model_vgg(input_4)
    
    x1 = Reshape((1, 1 ,128))(x1)
    x2 = Reshape((1, 1 ,128))(x2)
    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])
    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x1t = Lambda(lambda tensor : K.square(tensor))(x1)
    x2t = Lambda(lambda tensor : K.square(tensor))(x2)
    x3t = Lambda(lambda tensor : K.square(tensor))(x3)
    x4t = Lambda(lambda tensor : K.square(tensor))(x4)
    
    merged_sub1_fn        = Subtract()([x1, x2])
    merged_sub_square_fn  = Lambda(lambda tensor: K.square(tensor))(merged_sub1_fn)
    merged_sub1_vgg       = Subtract()([x3, x4])
    merged_sub_square_vgg = Lambda(lambda tensor: K.square(tensor))(merged_sub1_vgg)

    merged_sq_sub_fn  = Subtract()([x1t, x2t])
    merged_sq_sub_vgg = Subtract()([x3t, x4t])

    merged_mul1_fn  = Multiply()([x1, x2])
    merged_mul1_vgg = Multiply()([x3, x4])

    merged_sub_square_vgg = Conv2D(128, [1, 1])(merged_sub_square_vgg)
    merged_sq_sub_vgg     = Conv2D(128, [1, 1])(merged_sq_sub_vgg)
    merged_mul1_vgg       = Conv2D(128, [1, 1])(merged_mul1_vgg)

    merged = Concatenate(axis=-1)(
        [Flatten()(merged_sub_square_vgg), (merged_sub_square_fn),
         Flatten()(merged_sq_sub_vgg), (merged_sq_sub_fn),
         Flatten()(merged_mul1_vgg), (merged_mul1_fn)])
         
    merged = Dense(100, activation="relu")(merged)  ## use_bias=False
    merged = Dropout(0.1)(merged)
    merged = Dense(25, activation="relu")(merged)
    merged = Dropout(0.1)(merged)
    out = Dense(1, activation="sigmoid")(merged)

    model = Model([input_1, input_2, input_3, input_4], out)
    model.compile(loss="binary_crossentropy", metrics=['acc'], optimizer=Adam(0.0001))
    return model

def signed_sqrt(x):
    return K.sign(x)*K.sqrt(K.abs(x)+1e-9)

if __name__ == '__main__':
### dataset
    kd_id  = 6
    kd_num = 7
    data_folders_path = "/data/Dataset/FG_RFIW/Images/"
    
    all_families_kd = [] ##[[F0001,...], ...]
    families        = sorted(os.listdir(data_folders_path))
    length          = len(families)
    for i in range(kd_num):
        all_families_kd.append(families[math.floor(i / kd_num * length): math.floor((i + 1) / kd_num * length)])

    val_families = all_families_kd[kd_id]
    
    print("Val_kd:{}_{}".format(kd_id, len(val_families)))
    print(kd_id, val_families[0])
    
    all_images = [] # [/data/Dataset/FG_RFIW/Images/F0001/MID1/P00001_face0.jpg,....]
    for family in families:
        fm_mids = os.listdir(data_folders_path+family)
        fm_mids = [mid for mid in fm_mids if 'MID' in mid]
        for fmid in fm_mids:
            p1_dir = data_folders_path + family + '/'+ fmid
            p1_files = os.listdir(p1_dir)
            for p1_f in p1_files:
                all_images.append(p1_dir + '/' + p1_f)
    
    train_images = [x for x in all_images if x.split("/")[-3] not in val_families]
    train_person_to_images_map = defaultdict(list) ### {"F0001/MID1": [], }
    for x in train_images:
        train_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)
        
    val_images   = [x for x in all_images if x.split("/")[-3] in val_families]
    val_person_to_images_map = defaultdict(list)
    for x in val_images:
        val_person_to_images_map[x.split("/")[-3] + "/" + x.split("/")[-2]].append(x)

    relationships = [] ### [('F1018/MID6', 'F1018/MID7'),...]
    for f_id, f_dir in enumerate(families):
        fam_path = data_folders_path + f_dir 
        csv_file = pd.read_csv(fam_path + '/mid.csv')
        csv_df   = pd.DataFrame(csv_file)
        for i in range(1, len(csv_df)+1):
            col = csv_df[str(i)]
            for ind in range(i, len(col)) :
                if(col[ind] == 1 or col[ind] == 2 or col[ind] == 3 or col[ind] == 4 or col[ind] == 6 or col[ind] == 7 or col[ind] == 8):
                    p1 = f_dir + '/MID' + str(i)
                    p2 = f_dir + '/MID' + str(ind+1)
                    
                    p1_dir = fam_path + '/MID' + str(i)
                    p2_dir = fam_path + '/MID' + str(ind+1)
                  
                    if(os.path.exists(p1_dir) and os.path.exists(p2_dir)):
                        p1_files = os.listdir(p1_dir)
                        p2_files = os.listdir(p2_dir)
                        
                        if(len(p1_files)!=0 and len(p2_files) != 0):
                            relationships.append((p1,p2))
    print(len(relationships))

    train_relationships = [x for x in relationships if x[0].split("/")[0] not in val_families]
    val_relationships   = [x for x in relationships if x[0].split("/")[0] in val_families]

###Model
    facenet_path  = 'facenet_keras.h5'
    model_facenet = load_model(facenet_path)
    for layer in model_facenet.layers[:-3]:
        layer.trainable = True
        
    model_vgg = VGGFace(model='resnet50', include_top=False)
    for layer in model_vgg.layers[:-3]:
        layer.trainable = True

    IMG_SIZE_FN  = 160
    IMG_SIZE_VGG = 224

    save_file_path = "RKd" +str(kd_id)+ "/p1_f10025_300_aug_db_re50_{val_acc:.4f}_{epoch:03d}.h5"

    checkpoint        = ModelCheckpoint(save_file_path, monitor='val_acc', verbose=1, save_best_only=False, mode='auto', period=1)
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=15, verbose=1)
    callbacks_list    = [checkpoint, reduce_on_plateau]

    Rfiw_model = RFIW_Model2() 
    Rfiw_model.fit_generator(gen_train(train_relationships, train_person_to_images_map, batch_size=24), use_multiprocessing=True,
        validation_data=gen_val(val_relationships, val_person_to_images_map, batch_size=24), epochs=120, verbose=1,
        workers = 6, callbacks=callbacks_list, steps_per_epoch=200, validation_steps=100)
