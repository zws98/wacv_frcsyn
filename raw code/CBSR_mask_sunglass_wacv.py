# coding=utf-8
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import torch
from PIL import Image
import cv2
import numpy as np
import random
import caffe_pb2
import os
import os.path
import six
import string
import sys
#import ipdb
from torchvision import utils as vutils
from enum import Enum
from augmenter import Augmenter
import torch.multiprocessing as multiprocessing
def datum_to_array(datum):
    if(datum.encoded):
        data_encode = np.fromstring(datum.data,dtype=np.uint8)
        img_data = cv2.imdecode(data_encode, cv2.IMREAD_UNCHANGED)
        return img_data
    else:
        if len(datum.data):
            return np.fromstring(datum.data, dtype=np.uint8).reshape(
                    datum.channels, datum.height, datum.width)
        else:
            return np.array(datum.float_data).astype(float).reshape(
                    datum.channels, datum.height, datum.width)
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

class CAMERA(Enum):
    ID = 0
    SPOT = 1
    GENERAL = -1

class Record():
    def __init__(self, lmdb_idx=0):
        self.keys_id = []
        self.keys_spot = []
        self.keys_all = []
        self.next_iter = 0
        self.lmdb_idx = lmdb_idx

    # def get_keysall_sequential(self):
    #     it = iter(self.keys_all)
    #     return it.next()

class Record_merge():
    def __init__(self):
        self.keys_id = []
        self.keys_spot = []
        self.keys_all = []
        self.next_iter = 0


def default_filelist_reader(filelist, offset=0, lmdb_idx=0):
    with open(filelist, 'r') as file:
        all_lines = file.readlines()
        id_record_dict = {}
        ids = []

        for line in all_lines:
            key, label, camera = line.strip().split(' ')
            label_now = int(label) + offset
            if id_record_dict.has_key(label_now):
                record_now = id_record_dict[label_now]
            else:
                id_record_dict[label_now] = Record(lmdb_idx=lmdb_idx)
                record_now = id_record_dict[label_now]
                ids.append(label_now)

            if int(camera) == CAMERA.ID.value:
                record_now.keys_id.append(key)
                record_now.keys_all.append(key)
            elif int(camera) == CAMERA.SPOT.value:
                record_now.keys_spot.append(key)
                record_now.keys_all.append(key)
            elif int(camera) == CAMERA.GENERAL.value:
                record_now.keys_spot.append(key)
                record_now.keys_all.append(key)
            else:
                print('ERROR: Unknow camera type:', camera)

    return ids, id_record_dict

def default_filelist_reader_imgWise(filelist, offset=0,offset_max=0, lmdb_idx=0):
    with open(filelist, 'r') as file:

        all_lines = file.readlines()

        key_label_camera_list = []
        ids = set()


        for line in all_lines:
            key, label, camera = line.strip().split(' ') #

            label_now = int(label) + offset_max
            ids.add(label_now)
            key_label_camera_list.append((lmdb_idx, key, label_now)) #, int(camera)


    return list(ids), key_label_camera_list

def default_filelists_reader(filelists):
    offset = 0
    id_record_dict = {}
    ids = []
    filelist_count = 0
    for filelist in filelists:
        assert os.path.isfile(filelist)
        ids_temp, id_record_dict_temp = default_filelist_reader(filelist, offset=offset, lmdb_idx=filelist_count)
        ids.extend(ids_temp)
        # ipdb.set_trace()
        id_record_dict.update(id_record_dict_temp)
        offset = len(ids)
        filelist_count += 1
        print('INFO: ID offset:', offset, ' lmdb length:', len(ids_temp))
    return ids, id_record_dict

def imgWise_filelists_reader(filelists):
    offset = 0
    offset_max = 0
    # id_record_dict = {}
    ids = []
    key_label_camera_list = []
    filelist_count = 0
    for filelist in filelists:
        # print(filelist)
        assert os.path.isfile(filelist)
        ids_temp, key_label_camera_list_temp = default_filelist_reader_imgWise(filelist, offset=offset,offset_max=offset_max, lmdb_idx=filelist_count)
        ids.extend(ids_temp)
        # ipdb.set_trace()
        # id_record_dict.update(id_record_dict_temp)
        key_label_camera_list.extend(key_label_camera_list_temp)
        offset = len(ids)
        offset_max = max(ids)+1
        filelist_count += 1
        print('INFO: ID offset:', offset, ' lmdb length:', len(key_label_camera_list_temp))

    return ids, key_label_camera_list

class RandomDoubleSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        temp = torch.randperm(len(self.data_source)).long()
        temp_np = temp.numpy().repeat(2)
        # ipdb.set_trace()
        temp2 = torch.from_numpy(temp_np).long()
        return iter(temp2)

    def __len__(self):
        return len(self.data_source) * 2


class CBSRLMDB_imgWise(data.Dataset):
    def __init__(self, db_paths, filelist_paths,dic, shuffle=True, transform=None, target_transform=None,
                 read_imgs_perid_method='random_all'):
        self.read_imgs_perid_methods = ['random_all', 'sequential', 'one_id_one_spot', 'weight']
        assert read_imgs_perid_method in self.read_imgs_perid_methods
        self.read_imgs_perid_method = read_imgs_perid_method
        import lmdb
        self.dbs = []
        self.db_paths = db_paths
        assert len(db_paths) == len(filelist_paths) # 1
        # print(filelist_paths)
        # print(type(filelist_paths))#['../../../../data2/train_data/emore/emore_data/filelists_emore_train.txt']
        # exit()

        for db_path in db_paths:# 就一次
            assert os.path.exists(db_path)

            temp_env = lmdb.open(db_path, max_readers=1, readonly=True, lock=False,
                                 readahead=False, meminit=False) # 生成一个lmdb数据库文件

            temp_txn = temp_env.begin(write=False) # 打开这个数据库文件
            self.dbs.append(temp_txn)
            print('INFO: Loaded lmdb:', db_path)
        # exit()
        self.ids, self.key_label_camera_list = imgWise_filelists_reader(filelist_paths)

        self.length = len(self.key_label_camera_list)
        if shuffle:
            random.shuffle(self.key_label_camera_list)

        self.transform = transform
        self.target_transform = target_transform
        self.dic = dic

        self.augmenter = Augmenter(0.2, 0.2, 0.2)

    def __getitem__(self, index):
        img, target = None, None
        raw = None

        lmdb_idx, key1,id_now = self.key_label_camera_list[index]#, camera
        raw1 = self.dbs[lmdb_idx].get(key1.encode())
        # print(key.encode())
        # exit()

        # method1: opencv
        datum1 = caffe_pb2.Datum()

        datum1.ParseFromString(raw1)

        imgbuf1 = np.frombuffer(datum1.data, dtype=np.uint8)
        img1 = cv2.imdecode(imgbuf1, cv2.IMREAD_UNCHANGED)


        img = np.array(img1)
        #img1 = cv2.resize(img1, (224, 224), interpolation=cv2.INTER_CUBIC)
        # if key1.split('_')[0] == 'ditie':
        #     img = np.array(img1)
        # else:
        if random.random() > 0.8:
            try:
                colors_list = [(230, 216, 173), (0, 0, 0), (255, 255, 255)]
                color_select = colors_list[np.random.randint(0, 3)]  #
                pts = np.array(self.dic[key1.encode()], np.int)

                shape_idx = np.random.randint(0, 6)
                # shape_idx = 5
                if shape_idx == 0:
                    pts = np.clip(pts, 0, 111)
                    a = pts[23:].reshape(-1,2)
                    b_x = pts[0]
                    b_y = pts[1]
                    b = np.array([b_x, b_y])
                    b = b.reshape(1, 2)
                    c_x = pts[2]
                    c_y = pts[3]
                    c = np.array([c_x, c_y])
                    c = c.reshape(1, 2)
                    d_x = pts[4]
                    d_y = pts[5]
                    d = np.array([d_x, d_y])
                    d = d.reshape(1, 2)

                    t = np.concatenate((a, b, c, d), axis=0)
                if shape_idx == 1:
                    pts = np.clip(pts, 0, 111)
                    a = pts[23:].reshape(-1, 2)
                    b_x = pts[6]
                    b_y = pts[14]
                    b = np.array([b_x, b_y])
                    b = b.reshape(1, 2)
                    t = np.concatenate((a, b), axis=0)
                if shape_idx == 2:
                    pts = np.clip(pts, 0, 111)
                    a = pts[23:].reshape(-1, 2)
                    b_x = pts[7]
                    b_y = pts[8]
                    b = np.array([b_x, b_y])
                    b = b.reshape(1, 2)
                    t = np.concatenate((a, b), axis=0)
                if shape_idx == 3:
                    center_x = pts[9]
                    top = (pts[10] - pts[3]) / 2 + pts[3]
                    center_y = round((pts[11] - top) / 2 + top)
                    center = (center_x, center_y)
                    width = round((pts[12] - pts[13]) * 0.8 / 2)
                    height = round((pts[11] - top) / 2)
                if shape_idx == 4:
                    center_x1 = pts[17]
                    center_y1 = pts[18]
                    center1 = (center_x1, center_y1)
                    center_x2 = pts[19]
                    center_y2 = pts[20]
                    center2 = (center_x2, center_y2)

                    width = abs(center_x1-pts[21])
                    height = abs(center_y1-pts[22])
                if shape_idx == 5:
                    center_x1 = pts[17]
                    center_y1 = pts[18]
                    center1 = (center_x1, center_y1)
                    center_x2 = pts[19]
                    center_y2 = pts[20]
                    center2 = (center_x2, center_y2)

                    width = abs(center_x1 - pts[21])
                    height = abs(center_y1 - pts[22])
                    left_x1 = center_x1-width
                    left_y1 = center_y1-height
                    right_x1 = center_x1 + width
                    right_y1 = center_y1 + height
                    left1 = (left_x1, left_y1)
                    right1 = (right_x1, right_y1)

                    left_x2 = center_x2 - width
                    left_y2 = center_y2 - height
                    right_x2 = center_x2 + width
                    right_y2 = center_y2 + height
                    left2 = (left_x2, left_y2)
                    right2 = (right_x2, right_y2)


                # if shape_idx == 4:
                #     center_x = pts[9]
                #     top = pts[14]
                #     center_y = round((pts[11] - top) / 2 + top)
                #     center = (center_x, center_y)
                #     width = round((pts[12] - pts[13]) * 0.8 / 2)
                #     height = round((pts[11] - top) / 2)
                # if shape_idx == 5:
                #     iod = pts[15] - pts[16]
                #     top = round(pts[14] + 0.33 * iod)
                #     center_x = pts[9]
                #     center_y = round((pts[11] - top) / 2 + top)
                #     center = (center_x, center_y)
                #     width = round((pts[12] - pts[13]) * 0.8 / 2)
                #     height = round((pts[11] - top) / 2)
                if shape_idx <= 2:
                    img = cv2.fillPoly(img, [t], color_select, cv2.LINE_AA)  # (255, 239, 191) (28,28,28)
                    # cv2.imwrite('./vis_mask2.jpg', img)
                    # exit()
                if shape_idx == 3:
                    img = cv2.ellipse(img, center, (width, height), angle=0, startAngle=0, endAngle=360,
                                      color=color_select, thickness=-1, lineType=cv2.LINE_AA)

                if shape_idx == 4:
                    img = cv2.ellipse(img, center1, (width, height), angle=0, startAngle=0, endAngle=360,
                                      color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    img = cv2.ellipse(img, center2, (width, height), angle=0, startAngle=0, endAngle=360,
                                      color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
                    # cv2.imwrite('./vis_sun1.jpg', img)
                    # exit()
                if shape_idx == 5:
                    img = cv2.rectangle(img, left1, right1, color=(0, 0, 0), thickness = -1, lineType=cv2.LINE_AA, shift = 0)
                    img = cv2.rectangle(img, left2, right2, color=(0, 0, 0), thickness=-1, lineType=cv2.LINE_AA, shift=0)
                    # cv2.imwrite('./vis_sun2.jpg',img)
                    # exit()
            #
            except:
                img = np.array(img1)
                # cv2.imwrite('./vis.jpg', img)
                # exit()
            # exit()
        target = id_now
        img_rgb1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img1 = Image.fromarray(img_rgb1, 'RGB')
        img1 = self.augmenter.augment(img1)
        if self.transform is not None:
            img1 = self.transform(img1)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img1, target#, key1
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_paths + ')'

