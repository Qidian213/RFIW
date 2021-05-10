import os
import math
import pandas as pd
from random import choice, sample
from collections import defaultdict

if __name__ == '__main__':
    ### dataset
    kd_id  = 0
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
    
    val_images = [x for x in all_images if x.split("/")[-3] in val_families]
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

    val_relationships   = [x for x in relationships if x[0].split("/")[0] in val_families]

###
    p1 = []
    p2 = []
    batch_size = len(val_relationships) * 2

    ppl = list(val_person_to_images_map.keys())
    batch_tuples = val_relationships

    labels = [1] * len(batch_tuples)
    while len(batch_tuples) < batch_size:
        p1 = choice(ppl)
        p2 = choice(ppl)

        if p1 != p2 and (p1, p2) not in val_relationships and (p2, p1) not in val_relationships:
            batch_tuples.append((p1, p2))
            labels.append(0)

    for x in batch_tuples:
        if not len(val_person_to_images_map[x[0]]):
            print(x[0])

    X1 = [choice(val_person_to_images_map[x[0]]) for x in batch_tuples]
    X2 = [choice(val_person_to_images_map[x[1]]) for x in batch_tuples]
    index = list(range(len(labels)))

    dataframe = pd.DataFrame({'index': index, 'p1':X1, 'p2':X2, 'label': labels})
    dataframe.to_csv("val_kd" + str(kd_id) + ".csv", index=False,sep=',')