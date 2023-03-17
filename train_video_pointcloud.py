import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hashlib import md5
from multiprocessing import process
from operator import index
from pydoc import describe
import torch
import torch.cuda
import torch.optim
import torch.nn as nn
import torch.nn.functional as F

import json
import imageio
import os
from os import path
import time
import shutil
import gc
import math
import argparse
import pickle

import numpy as np

from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union
from loguru import logger
from multiprocess import Pool

import debug
from point_radiance.modules.model import CoreModel as PointSHNeRF
from point_radiance.dataloader.dataset import Data
from util.args import config_args
from torch.multiprocessing  import Queue, Process
from queue import Empty

from point_radiance.modules.utils import mse2psnr, \
    grad_loss, safe_path, set_seed

def extra_args(parser):
    config_util.define_common_args(parser)

    group = parser.add_argument_group("general")
    group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')

    group = parser.add_argument_group("optimization")
    group.add_argument('--n_iters', type=int, default=20, help='total number of iters to optimize for')
    group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
    group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
    group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

    group.add_argument('--print_every', type=int, default=20, help='print every')
    group.add_argument('--save_every', type=int, default=5,
                    help='save every x epochs')
    group.add_argument('--eval_every', type=int, default=1,
                    help='evaluate every x epochs')

    group.add_argument('--pretrained', type=str, default=None,
                    help='pretrained model')

    group.add_argument('--frame_start', type=int, default=1, help='train frame among [frame_start, frame_end]')  
    group.add_argument('--frame_end', type=int, default=30, help='train frame among [1, frame_end]')
    group.add_argument('--fps', type=int, default=30, help='video save fps')

    group.add_argument('--train_use_all', type=int, default=0 ,help='whether to use all image as training set')
    group.add_argument('--save_every_frame', action='store_true', default=False)
    group.add_argument('--dilate_rate_before', type=int, default=2, help="dilation rate for grid.links before training")
    group.add_argument('--dilate_rate_after', type=int, default=2, help=" dilation rate for grid.links after training")
    group.add_argument('--pilot_factor', type=int, default=1, help="dilation rate for grid.links after training")

    group.add_argument('--offset', type=int, default=250)

    group.add_argument('--compress_saving', action="store_true", default=False, help="dilation rate for grid.links")
    group.add_argument('--sh_keep_thres', type=float, default=1)
    group.add_argument('--sh_prune_thres', type=float, default=0.2)

    group.add_argument('--performance_mode', action="store_true", default=False, help="use perfomance_mode skip any unecessary code ")
    group.add_argument('--debug',  action="store_true", default=False,help="switch on debug mode")
    group.add_argument('--keep_rms_data',  action="store_true", default=False,help="switch on debug mode")

    group.add_argument('--remove_lowvc_area',  action="store_true", default=False,help="use grad ratio")
    group.add_argument('--use_grad_ratio',  action="store_true", default=False,help="use grad ratio")
    group.add_argument('--apply_error_cache',  action="store_true", default=False,help="use grad ratio")
    group.add_argument('--apply_view_importance',  action="store_true", default=False,help="use grad ratio")
    group.add_argument('--apply_rgb_diff',  action="store_true", default=False,help="use grad ratio")
    group.add_argument('--apply_narrow_band',  action="store_true", default=False,help="apply_narrow_band")
    group.add_argument('--render_all',  action="store_true", default=False,help="rqender all camera in sequence")
    group.add_argument('--save_delta',  action="store_true", default=False,help="save delta in compress saving")

    group.add_argument('--gpu_id', type=int, default=-1, help='ID of desired GPU')
    group.add_argument('--batch_size', type=int, default=1, help='# of views to train on')
    group.add_argument('--init_epochs', type=int, default=-1, help='# of epochs to train base model with')

    return parser

args = config_args(extra_args)
args = args.parse_args()
device = "cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"

summary_writer = SummaryWriter(args.train_dir)

torch.manual_seed(20200823)
np.random.seed(20200823)

class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.dataname = args.dataname
        self.logpath = args.basedir
        self.outpath = safe_path(os.path.join(self.logpath, 'output'))
        self.weightpath = safe_path(os.path.join(self.logpath, 'weight'))
        self.imgpath = safe_path(os.path.join(self.outpath, 'images'))
        self.imgpath = safe_path(os.path.join(self.imgpath, '{}'.format(self.dataname)))
        self.logfile = os.path.join(self.outpath, 'log_{}.txt'.format(self.dataname))
        self.logfile = open(self.logfile, 'w')
        self.loss_fn = torch.nn.MSELoss()
        self.lr1, self.lr2, self.lr3  = args.lr1, args.lr2, args.lr3
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        self.set_optimizer(self.lr1, self.lr2)
        self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.i_split = self.model.i_split
        self.imagesgt_train = self.imagesgt
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r)))
        self.training_time = 0

        self.init_epochs = args.init_epochs
        self.batch_size = args.batch_size

    def set_onlybase(self):
        self.model.onlybase = True
        self.set_optimizer(self.lr1,self.lr2)
    
    def remove_onlybase(self):
        self.model.onlybase = False
        self.set_optimizer(self.lr3,self.lr2)

    def set_optimizer(self, lr1=3e-3, lr2=8e-4):
        sh_list = [name for name, params in self.model.named_parameters() if 'sh' in name]
        sh_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in sh_list,
                                self.model.named_parameters()))))
        other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in sh_list,
                                self.model.named_parameters()))))
        optimizer = torch.optim.Adam([
            {'params': sh_params, 'lr': lr1},
            {'params': other_params, 'lr': lr2}])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.lrexp, -1)
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        return None

    def test(self):
        with torch.no_grad():
            loss_all, psnr_all = [], []
            test_ids = self.i_split[2]
            for id in test_ids:
                images = self.model(id)
                loss = self.loss_fn(images[0], self.imagesgt[id])
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
                if visual:
                    pred = images[0, ..., :3].detach().cpu().data.numpy()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    # set background as white for visualization
                    mask = self.masks[id].cpu().data.numpy()
                    pred = pred*mask+1-mask
                    gt = gt*mask+1-mask
                    img_gt = np.concatenate((pred,gt),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join(self.imgout_path,
                            'img_{}_{}_{:.2f}.png'.format(self.dataname, id, mse2psnr(loss).item())))
            # loss_e = torch.stack(loss_all).mean().item()
            # psnr_e = torch.stack(psnr_all).mean().item()
            # info = '-----eval-----  loss:{:.3f} psnr:{:.3f}'.format(loss_e, psnr_e)
            # print(info)
            return psnr_e

    def train(self):
        max_psnr = 0.
        start_time = time.time()
        for epoch in range(epoch_n):
            loss_all, psnr_all = [], []
            ids = self.i_split[0]
            for id in tqdm(ids):
                images = self.model(id)
                loss = self.loss_fn(images[0], self.imagesgt_train[id])
                loss = loss + self.lr_s * grad_loss(images[0], self.imagesgt_train[id])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
            self.lr_scheduler.step()
            # loss_e = torch.stack(loss_all).mean().item()
            # psnr_e = torch.stack(psnr_all).mean().item()
            # info = '-----train-----  epoch:{} loss:{:.3f} psnr:{:.3f}'.format(epoch, loss_e, psnr_e)
            # print(info)
            # self.logfile.write(info + '\n')
            # self.logfile.flush()
            psnr_val = self.test(self.i_split[1], False)
            if psnr_val > max_psnr:
                max_psnr = psnr_val
        
        torch.save(self.model.state_dict(), os.path.join(
                 self.weightpath,'model_{}.pth'.format(self.dataname)))

    def solve_initial_frame(self):
        self.set_onlybase()
        self.train(epoch_n=args.init_epochs)
        self.remove_onlybase()
        self.train(epoch_n=args.init_epochs)
        for i in range(args.refine_n):
            trainer.model.remove_out()
            trainer.model.repeat_pts()
            trainer.train(epoch_n=args.init_epochs)

def fetch_dataset(frame_idx):
    args.frame_id = frame_idx
    dataset = Data(args)
    memitem = dataset.initpc()

    return memitem

def pre_fetch_datasets(frame_idx):
    while True:
        try:
            logger.debug(f'Waiting for new frame... (Queue Size: {frame_idx_queue.qsize()})')
            frame_idx = frame_idx_queue.get(block=True,timeout=60)
            logger.debug(f'Frame received. (Queue Size: {frame_idx_queue.qsize()})')
        except Empty:
            logger.debug('Ending data prefetch process.')
            return 
        
        logger.debug(f"Finished loading frame: {frame_idx}")
        dset_queue.put(fetch_dataset(frame_idx))

frame_idx_queue = Queue()
dset_queue = Queue()


if args.pretrained is not None:
    print("LOAD MODEL")
else:
    base_memitem = fetch_dataset(args.frame_start)
    base_model = PointSHNeRF(args, base_memitem)
trainer = Trainer(args, base_model)
print("HERE")
trainer.solve_initial_frame()

pre_fetch_process = Process(target=pre_fetch_datasets)
pre_fetch_process.start()
prefetch_factor = 3
for i in range(prefetch_factor):
    frame_idx_queue.put(i + args.frame_start)

train_frame_num = 0
global_step_base = 0
frames = []
psnr_list = []

# for frame_idx in range(args.frame_start, args.frame_end) :
#     # dset = dset_iter[frame_idx - args.frame_start]
#     dset = dset_queue.get(block=True)

#     if frame_idx + prefetch_factor < args.frame_end:
#         frame_idx_queue.put(frame_idx + prefetch_factor)

#     frame, psnr = finetune_one_frame(frame_idx, global_step_base, dset)
#     frames.append(frame)
#     psnr_list.append(psnr)
#     if args.save_every_frame:
#         os.makedirs(os.path.join(args.train_dir,"ckpts"))
#         grid.save(os.path.join(args.train_dir,"ckpts",f'{frame_idx:04d}.npz'))

#     global_step_base += args.n_iters
#     train_frame_num += 1

# logger.critical(f'average psnr {sum(psnr_list)/len(psnr_list):.4f}')
# for i in range(len(psnr_list)):
#     logger.critical(f'Frame {i} psnr: {psnr_list[i]}')

# with open(os.path.join(args.train_dir, 'grid_delta_pilot', 'psnr_data.pkl'), "wb") as fp:
#     pickle.dump(psnr_list, fp)


# if train_frame_num:
   
#     tag = os.path.basename(args.train_dir) 
#     vid_path = os.path.join(args.train_dir, tag+'_pilot.mp4')
#     imageio.mimwrite(vid_path, frames, fps=args.fps, macro_block_size=8)
#     logger.info('video write to', vid_path)

#     grid.density_rms = torch.zeros([1])
#     grid.sh_rms = torch.zeros([1])
#     grid.save(os.path.join(args.train_dir, 'ckpt.npz'))

# pre_fetch_process.join()
# pre_fetch_process.close()