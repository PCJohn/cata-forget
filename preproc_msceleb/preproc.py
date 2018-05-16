import cv2
import os
import shutil

import tsv_extractor

path_to_tsv_file = '/home/prithvi/dsets/MSCeleb/FaceImageCroppedWithAlignment.tsv'
path_to_preproc_imgs = '/home/prithvi/dsets/MSCeleb/MSCeleb'
path_to_clean_list = '/home/prithvi/dsets/MSCeleb/MS-Celeb-1M_clean_list.txt'
n_ppl = 1000
min_img_per_celeb = 90

def keep_many(base_path):
    max_img_per_celeb = 20000
    base_set = os.listdir(base_path)
    for p in base_set:
        pth = os.path.join(base_path,p)
        if os.path.split(pth.strip())[-1] == 'bboxes.txt': #'/home/prithvi/dsets/MSCeleb/MSCeleb/bboxes.txt':
            continue
        t = os.listdir(pth)
        if (len(t) < min_img_per_celeb) | (len(t) > max_img_per_celeb):
            shutil.rmtree(pth)

def clean():
    if not os.path.exists(clean_path):
        os.mkdir(clean_path)
    with open(clean_list,'r') as f:
        lines = f.readlines()
    for line in lines[:]:
        pth,pid = line.split(' ')
        pid = pid.strip()
        pth = pth.strip()
        pth = os.path.join(base,pth)
        fname = os.path.split(pth)[-1]
        m = os.path.split(os.path.split(pth)[-2])[-1]
        mi = os.path.split(pth)[-1]
        mi = mi.replace('-','_',1)
        pth = os.path.join(''.join(os.path.split(pth)[:-1]),mi)
        #if m.startswith('m.01_0d'):
        #    print '>>>',os.path.join(''.join(os.path.split(pth)[:-1]),mi)
        #print fname
        #print pid
        if os.path.exists(pth):
            img = cv2.imread(pth)
            #cv2.imshow(str(pid),img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            save_path = os.path.join(clean_path,str(pid))
            print '>>>',pth
            print '\t',save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(os.path.join(save_path,fname),img)
            os.remove(pth)

#tsv_path = path_to_tsv_file
clean_list = path_to_clean_list
clean_path = path_to_preproc_imgs #'/home/prithvi/dsets/MSCeleb/MSCeleb'
raw_path = '/home/prithvi/dsets/MSCeleb/raw'
base = raw_path
tsv_extractor.extract(path_to_tsv_file,raw_path,n_ppl)
clean()
keep_many(clean_path)
shutil.rmtree(raw_path)
