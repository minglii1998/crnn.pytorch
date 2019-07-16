import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import os
import glob

import models.crnn as crnn

'''
This code can only be used to test those pictures in folders.
And the label is on the name.
While using, 
test_path means the path that your pictures are in.
model_path maens the path where the trained model is .
only these two need to be modified.
'''

# get pic and label
test_path = r"/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1"
g = os.walk(test_path)
test_list = []
label_list = []

for path,dir_list,file_list in g:  
    new_list = glob.glob(os.path.join(path, '*.jpg'))
    test_list = test_list + new_list

for item in test_list:
    label_list.append(item.split('_')[1].lower())


pic_label_list = zip(test_list,label_list)
pic_count = 0
right_count = 0

model_path = './expr/netCRNN_1_99500.pth'
#img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

for img_path,label in pic_label_list:
    pic_count += 1

    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(img_path).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    str_pred = "".join(sim_pred)

    if (str_pred == label):
        right_count+=1
    else:
        print('predicted: %-20s,real: %-20s,path: %s ' 
        % (sim_pred, label, img_path))

print ("number of tested: %d" % pic_count)
print ("number of right: %d" % right_count)
print ("Accuracy : %f" % (right_count/pic_count))
