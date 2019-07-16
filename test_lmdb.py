import numpy as np
import os
import dataset
import utils
import torch
from torch.autograd import Variable
import lmdb
import models.crnn as crnn
import six
import sys
from PIL import Image

root = '/mnt/data_share/text_recog/data_lmdb_release/evaluation/IC03_860'
model_path = './expr/netCRNN_1_99500.pth'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
pic_count = 0
right_count = 0

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

env = lmdb.open(
    root,
    max_readers=1,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False)

with env.begin(write=False) as txn:
    nSamples = int(txn.get('num-samples'.encode()))
    print(nSamples)

for Index in range(1,nSamples+1):

    pic_count += 1

    img_key = 'image-%09d' % Index
    label_key = 'label-%09d' % Index

    with env.begin(write=False) as txn:
        label_key = 'label-%09d' % Index
        label = txn.get(label_key.encode()).decode()

        imgbuf = txn.get(img_key.encode())
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = Image.open(buf).convert('L')

    transformer = dataset.resizeNormalize((100, 32))
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
        print('predicted: %-20s,real: %-20s,key: %s ' 
        % (sim_pred, label, img_key))


print ("number of tested: %d" % pic_count)
print ("number of right: %d" % right_count)
print ("Accuracy : %f" % (right_count/pic_count))