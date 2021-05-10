import os
import cv2
import tqdm
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import load_model
from keras_vggface.utils import preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

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
    
def signed_sqrt(x):
    return K.sign(x)*K.sqrt(K.abs(x)+1e-9)

def chunker(seq, size=64):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

if __name__ == '__main__':
### dataset
    IMG_SIZE_FN  = 160
    IMG_SIZE_VGG = 224

### model
    model_path = 'Kd6/p1_f10025_300_aug_db_re50_0.7888_037.h5'
    RFModel    = load_model(model_path, custom_objects={'signed_sqrt': signed_sqrt})
  
    X1 = ['/data/Dataset/FG_RFIW/Images/F0001/MID4/P00006_face1.jpg', '/data/Dataset/FG_RFIW/Images/F0005/MID1/P00053_face1.jpg']
    X2 = ['/data/Dataset/FG_RFIW/Images/F0001/MID2/P00002_face1.jpg', '/data/Dataset/FG_RFIW/Images/F0001/MID2/P00002_face1.jpg']
    X1_FN  = np.array([read_img_fn(x) for x in X1])
    X1_VGG = np.array([read_img_vgg(x) for x in X1])

    X2_FN  = np.array([read_img_fn(x) for x in X2])
    X2_VGG = np.array([read_img_vgg(x) for x in X2])
    
    preds  = RFModel.predict([X1_FN, X2_FN, X1_VGG, X2_VGG]).ravel().tolist()
    print(preds)

# if __name__ == '__main__':
# ### dataset
    # IMG_SIZE_FN  = 160
    # IMG_SIZE_VGG = 224

# ### model
    # model_path = 'Kd6/p1_f10025_300_aug_db_re50_0.7888_037.h5'
    # RFModel    = load_model(model_path, custom_objects={'signed_sqrt': signed_sqrt})

# ### inference
    # indexs  = []
    # results = []
    # labels  = []

    # valfile = pd.read_csv('Kds/val_kd6.csv')
    # valdata = pd.DataFrame(valfile)
    # batchs  = chunker(list(range(len(valdata))))

    # #for batch in tqdm(chunker(file.index.values)):
    # for step, batch in enumerate(batchs):
        # print(f"{step}/{len(batchs)}")
        # index = [valdata['index'][ind] for ind in batch]
        # label = [valdata['label'][ind] for ind in batch]
        # X1    = [valdata['p1'][ind] for ind in batch]
        # X2    = [valdata['p2'][ind] for ind in batch]
        
        # X1_FN  = np.array([read_img_fn(x) for x in X1])
        # X1_VGG = np.array([read_img_vgg(x) for x in X1])

        # X2_FN  = np.array([read_img_fn(x) for x in X2])
        # X2_VGG = np.array([read_img_vgg(x) for x in X2])
        
        # preds  = RFModel.predict([X1_FN, X2_FN, X1_VGG, X2_VGG]).ravel().tolist()
        
        # results += preds
        # indexs  += index
        # labels  += label

    # dataframe = pd.DataFrame({'index':indexs,'label':results, 'gt': labels})
    # dataframe.to_csv("results/val_" + model_path.split('/')[-1].replace('h5', 'csv'),index=False,sep=',')

# ### compute acc 
    # Corect_Num = 0
    # Pos_Th     = 0.5
    # for prd, gt in zip(results, labels):
        # if((prd>=0.5 and gt==1) or (prd<0.5 and gt==0)):
            # Corect_Num += 1
    # print(f'Val Acc: {Corect_Num/len(labels):.4f}')
