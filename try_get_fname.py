import scipy.io as sio
import numpy as np
import glob
import os
import cv2
from cv2 import cv2 as cv
import h5py

# 遍历文件夹下的所有目录
test_path = r"/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1"
g = os.walk(test_path)
test_list = []
label_list = []

for path,dir_list,file_list in g:  
    new_list = glob.glob(os.path.join(path, '*.jpg'))
    test_list = test_list + new_list

for item in test_list:
    label_list.append(item.split('_')[1].lower())

print(label_list)




# 直接读文件中的图片的
'''
test_path = '/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1/1/'

test_list = glob.glob(os.path.join(test_path, '*.jpg'))
label_list = []

for item in test_list:
    label_list.append(item.split('_')[1].lower())
    
print (label_list)
'''

# 下面这些是之前用来读.mat文件的，现在先不管了
'''
data=h5py.File(test_label_path)
k = list(data.keys())
ds = data['digitStruct']
print(list(ds))
#print(list(ds['bbox']))
print(list(data[ds['bbox'][0,0]]))

#print(list(data[data[ds['bbox'][0,0]][0,0]]))



test = data['digitStruct/name']
st = test[0][0]
obj = data[st]
print(list(obj))


struArray = data['digitStruct']

print(struArray)

print(data.values())

print(data['digitStruct'].values())
'''
