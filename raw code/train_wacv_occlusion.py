from __future__ import print_function

import torch.optim as optim
import torchvision.transforms as transforms
from CBSR_mask_sunglass_wacv import *
from funcs import *
from resnetSEIR import *
from torch import distributed as dist
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from margin.AdaFace import AdaFace


# from build import build_model
#from backbone.gkd_adaptive_93 import GKD
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

from torchvision import utils as vutils
import time
lmk_candidate = [150,59,102,103,140,7,106,120,121,104,105,33,52,12,107,160,142,208,209,210,211,102,81]

dic = {}
lmks_path = '/datazws/WACV_DATA/filelists_casia_train_lmk.txt'
with open(lmks_path, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        key = line.rstrip().split(' ')[0]
        value = [int(i) for i in list(line.rstrip().split(' ')[2:])]
        value_new = []
        for i in lmk_candidate:
            value_new.extend([value[i]])
        value_new.extend(value[6:60])
        value_new = np.array(value_new).astype(np.uint8)
        dic.setdefault(key.encode(), value_new)
# lmks_path = '/datazws/WACV_DATA/filelists_ffhq_train_lmk.txt'
# with open(lmks_path, 'r') as fr:
#     lines = fr.readlines()
#     for line in lines:
#         key = line.rstrip().split(' ')[0]
#         value = [int(i) for i in list(line.rstrip().split(' ')[2:])]
#         value_new = []
#         for i in lmk_candidate:
#             value_new.extend([value[i]])
#         value_new.extend(value[6:60])
#         value_new = np.array(value_new).astype(np.uint8)
#         dic.setdefault(key.encode(), value_new)
lmks_path = '/datazws/WACV_DATA/filelists_dcface_train_lmk.txt'
with open(lmks_path, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        key = line.rstrip().split(' ')[0]
        value = [int(i) for i in list(line.rstrip().split(' ')[2:])]
        value_new = []
        for i in lmk_candidate:
            value_new.extend([value[i]])
        value_new.extend(value[6:60])
        value_new = np.array(value_new).astype(np.uint8)
        dic.setdefault(key.encode(), value_new)
# lmks_path = '/datazws/WACV_DATA/filelists_gandiff_train_lmk.txt'
# with open(lmks_path, 'r') as fr:
#     lines = fr.readlines()
#     for line in lines:
#         key = line.rstrip().split(' ')[0]
#         value = [int(i) for i in list(line.rstrip().split(' ')[2:])]
#         value_new = []
#         for i in lmk_candidate:
#             value_new.extend([value[i]])
#         value_new.extend(value[6:60])
#         value_new = np.array(value_new).astype(np.uint8)
#         dic.setdefault(key.encode(), value_new)
# lmks_path = '/datazws/idiff_lmk.txt'
# with open(lmks_path, 'r') as fr:
#     lines = fr.readlines()
#     for line in lines:
#         value=[]
#         key = line.rstrip().split('\t')[0]
#         l = line.rstrip().split('\t')[1]
#
#         for i in l.split(' ')[0:]:
#             value.append(int(i))
#
#
#         value_new = []
#         for i in lmk_candidate:
#             value_new.extend([value[i]])
#         value_new.extend(value[6:60])
#         value_new = np.array(value_new).astype(np.uint8)
#         dic.setdefault(key.encode(), value_new)
def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
logger = Logger('./log/dcface_dcface_warm2_os2_cosface_IR100.log')####hongwai
setup_seed(seed=213+local_rank, cuda_deterministic=True)
num_epoch = 23#25
lr = 0.1

warmup_epoch = 2
min_lr = 2.5e-7
warmup_lr = 2.5e-7

batch_size = 160
start = 0
start_epoch = 0
total_batch = 0



student = SEResNet_IR(100, num_classes = 93431)# 93431 # 93232+6326+14656

# student.fc = MarginCosineProduct(512, 20652, m=0.4, scale=64) # 10572+3726
# student.fc = MarginCosineProduct(512, 67889, m=0.4, scale=64) # 10572+57316
# student.fc = MarginCosineProduct(512, 77889, m=0.4, scale=64) # 10572+57316+10000
# student.fc = MarginCosineProduct(512, 87969, m=0.4, scale=64) # 10572+57316+10000+10080
student.fc = AdaFace(embedding_size=512,
                       classnum=20572,#20652,#25000,
                       m=0.4,
                       h=0.333,
                       s=64,
                       t_alpha=0.01,
                       )

student = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student).to(device)

student = torch.nn.parallel.DistributedDataParallel(student,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,find_unused_parameters=True)


optimizer = optim.SGD(student.parameters(),lr=lr, momentum=0.9, weight_decay = 5e-4, nesterov=True)



db_paths = [

'/media/ssd4/casia.lmdb',
# '/media/ssd4/ffhq.lmdb',
'/media/ssd4/dcface.lmdb',
# '/media/ssd4/gandiff.lmdb',
# '/media/ssd4/idiff.lmdb'

            ] #'/media/ssd4/webface_mask.lmdb'
filelist_paths = [
'/media/ssd4/casia_train_cleansing_sim0.3_os2.txt',
#'/datazws/WACV_DATA/casia_train_cleansing_sim0.3_os2.txt',# '/datazws/WACV_DATA/filelists_casia_train.txt',
# '/data1/filelists_ffhq_train_casia_each_0.63_os2.txt', # '/data1/filelists_ffhq_train_cluster.txt' #
#  '/datazws/WACV_DATA/dcface_train_cleansing_sim0.3.txt', # '/datazws/WACV_DATA/filelists_dcface_train_casia_each_0.63.txt',
#     '/datazws/WACV_DATA/gandiff_train_cleansing_sim0.3.txt'
# '/data1/filelists_idiff_train_casia_each_0.5.txt'
# '/datazws/WACV_DATA/filelists_gandiff_train_casia_each_0.6.txt'
'/media/ssd4/filelists_dcface_train_casia_each_0.63.txt',
# '/datazws/WACV_DATA/filelists_dcface_train_casia_each_0.63.txt',
 #'/datazws/WACV_DATA/dcface_train_cleansing_sim0.3_50.txt', # '/datazws/WACV_DATA/filelists_dcface_train_casia_each_0.63.txt',
    # '/datazws/WACV_DATA/gandiff_train_cleansing_sim0.3.txt',
# '/data1/filelists_idiff_train_casia_each_0.5_20.txt'

                  ]#'/data1/filelists_webface_mask_train.txt'
dataset = CBSRLMDB_imgWise(db_paths, filelist_paths,dic=dic,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.Resize(112),
                        transforms.RandomHorizontalFlip(),
                        #transforms.Grayscale(num_output_channels=1),
                        transforms.ToTensor(),
                        #transforms.Normalize([0.5], [0.5])
                        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
                     ])
                           )


num_tasks = dist.get_world_size()
global_rank = dist.get_rank()
sampler_train = DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
train_loader= DataLoader(
        dataset, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )
n_iter_per_epoch = len(train_loader)
num_steps = int(num_epoch * n_iter_per_epoch)
warmup_steps = int(warmup_epoch * n_iter_per_epoch)

scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            # t_mul=1.,
            lr_min=min_lr,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
criterion = nn.CrossEntropyLoss()
criterion.to(device)


batch_time = AverageMeter()
load_time = AverageMeter()
top1 = AverageMeter()
running_loss = AverageMeter()


for epoch in range(start,num_epoch):  #

    batch_time = AverageMeter()
    load_time = AverageMeter()
    top1 = AverageMeter()

    running_loss = AverageMeter()

    end_time = time.time()
    student.train()
    maxid = 0
    for batch_idx, (images, labels) in enumerate(train_loader):

        load_time.update(time.time() - end_time)
        # print(labels)
        # exit()
        images = images.to(device) # Bx112x112x3

        labels = labels.to(device) # Bx1 [0,cls]
        # images = images.detach().cpu()

        # compute outputs
        outputs_student = student(images,labels)
        # continue

        loss = criterion(outputs_student,labels) #

        prec1, = compute_accuracy(outputs_student, labels, topk=(1,))

        top1.update(prec1.item())

        # backward
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        scheduler.step_update(epoch * n_iter_per_epoch + batch_idx)
        running_loss.update(loss.item())
        #running_loss_ce.update(loss_ce.item())

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        total_batch += 1
        lr_now = optimizer.param_groups[0]['lr']

        if batch_idx % 100 == 0:
            if local_rank == 0:
                logger("Epoch-Iter [%d/%d][%d/%d] Time_tot/load [%f][%f] lr [%g] loss [%f] Top1 [%f]"%(
                    epoch + 1, num_epoch , batch_idx, len(train_loader), batch_time.avg,
                    load_time.avg, lr_now, running_loss.avg, top1.avg))#,


    # scheduler.step()
    if (epoch + 1 + start_epoch) % 1 == 0:
        if local_rank == 0:
            # torch.save({'state_dict': student.module.state_dict(),  # model.module.state_dict()
            #             }, '/datazws/casia_ffhq_cluster_clean_dcface_0.63_gandiff_0.6_mask_0.2_intraclean_casia_0.3_dcface_0.3_gandiff_0.3_os2_adaface_IR100/train_cosinelr_IR100_casia_ffhq_cluster_clean_dcface0.63_gandiff_0.6_mask_0.2_intraclean_casia_0.3_dcface_0.3_gandiff_0.3_os2_adaface_epoch_' + str(
            #     epoch + 1 ) + '_' + str(local_rank) + '.tmp')
            torch.save({'state_dict': student.module.state_dict(),  # model.module.state_dict()
                        },
                       '/datazws/casia_dcface_os2_adafaug_v2_IR100/train_cosinelr_IR100_cosface_epoch_' + str(
                           epoch + 1) + '_' + str(local_rank) + '.tmp')

            print('save model')
