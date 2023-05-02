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
import torch.multiprocessing as mp

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
from PIL import Image

import debug
from point_radiance.modules.model import CoreModel as PointSHNeRF
from point_radiance.dataloader.dataset import Data
from util.args import config_args
from torch.multiprocessing  import Process, Queue
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

    group.add_argument('--frame_start', type=int, default=0, help='train frame among [frame_start, frame_end]')  
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
    group.add_argument('--init_epochs_base', type=int, default=20, help='# of epochs to train base model with')
    group.add_argument('--init_epochs_full', type=int, default=30, help='# of epochs to train base model with')
    group.add_argument('--f2f_epochs', type=int, default=30, help='# of epochs to train model frame to frame with')
    group.add_argument('--dpp_radius', type=float, default=1.75, help='Defines redius of dialation')
    group.add_argument('--dialation_thresh', type=float, default=0.1, help='Defines threshold for max points of dialation')
    group.add_argument("--double_steps", type=int, default=1, help='number of times to double pointcloud before refinement')
    group.add_argument("--init_model", type=str, default=None, help='Base model to use in experiments')
    group.add_argument("--save_init_model", action="store_true", default=False, help='Whether to save model being trained')
    group.add_argument("--name", type=str, default="untitled", help='Name of our experiment')
    
    group.add_argument("--origin_vert_thresh", type=float, default=0.05, help='Pruning threshold for origin vertices movement')
    group.add_argument("--origin_sh_thresh", type=float, default=0.025, help='Pruning threshold for origin vertices spherical harmonics')
    group.add_argument("--origin_sh_thresh_zero", type=float, default=0.06, help='Pruning threshold for origin vertices spherical harmonics moving to zero')
    group.add_argument("--dialated_vert_thresh", type=float, default=0.1, help='Pruning threshold for dialated vertices movement')
    group.add_argument("--dialated_sh_thresh", type=float, default=0.22, help='Pruning threshold for dialated vertices spherical harmonics')

    group.add_argument("--disable_origin_movement", action="store_true", default=False, help='Disables movement of origin points while training')
    group.add_argument("--disable_origin_sh", action="store_true", default=False, help='Disables spherical harmonics of origin points while training')

    return parser


class Trainer():
    def __init__(self, args, model, device):
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
        self.lr1, self.lr2, self.lr3, self.lr4, self.lr5, self.lr6  = args.lr1, args.lr2, args.lr3, args.lr4, args.lr5, args.lr6
        self.lrexp, self.lr_s = args.lrexp, args.lr_s
        self.set_optimizer(self.lr1, self.lr2)
        self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.i_split = self.model.i_split
        self.imagesgt_train = self.imagesgt
        self.imgout_path = safe_path(os.path.join(self.imgpath,
                        'v2_{:.3f}_{:.3f}'.format(args.data_r, args.splatting_r)))
        self.training_time = 0

        self.init_epochs_base = args.init_epochs_base
        self.init_epochs_full = args.init_epochs_full
        self.batch_size = args.batch_size

    def set_onlybase(self):
        self.model.onlybase = True
        self.set_optimizer(self.lr1,self.lr2)
    
    def remove_onlybase(self):
        self.model.onlybase = False
        self.set_optimizer(self.lr3,self.lr4)

    def set_f2f_train(self):
        self.model.onlybase = False
        self.set_optimizer(self.lr5,self.lr6)

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

    def test(self, visual=False):
        with torch.no_grad():
            loss_all, psnr_all = [], []
            test_ids = np.array(self.i_split[2]).reshape(-1, 1)
            for id in test_ids:
                images = self.model(id)
                loss = self.loss_fn(images[0], self.imagesgt[id[0]])
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
                if visual:
                    pred = images[0].detach().cpu().data.numpy()
                    gt = self.imagesgt[id].detach().cpu().data.numpy()
                    img_gt = np.concatenate((pred,gt[0]),1)
                    img_gt = Image.fromarray((img_gt*255).astype(np.uint8))
                    img_gt.save(os.path.join("test",'img_{}_{}_{}.png'.format(self.args.name, id, self.model.frame_id)))
                    
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            eval_message = '-----eval-----  loss:{:.3f} psnr:{:.3f}'.format(loss_e, psnr_e)
            logger.critical(eval_message)

            self.training_time += 1
            return psnr_e

    def train(self, epoch_n=20):
        max_psnr = 0.
        for epoch in range(epoch_n):
            loss_all, psnr_all = [], []
            np.random.shuffle(self.i_split[0])
            ids = np.array(self.i_split[0]).reshape(-1, self.batch_size)
            for id_batch in tqdm(ids):
                images = self.model(id_batch)
                loss = self.loss_fn(images, self.imagesgt_train[id_batch])
                loss = loss + self.lr_s * grad_loss(images, self.imagesgt_train[id_batch])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_all.append(loss)
                psnr_all.append(mse2psnr(loss))
            self.lr_scheduler.step()
            loss_e = torch.stack(loss_all).mean().item()
            psnr_e = torch.stack(psnr_all).mean().item()
            eval_message = '-----train-----  epoch:{} loss:{:.3f} psnr:{:.3f}'.format(epoch, loss_e, psnr_e)
            logger.critical(eval_message)
            psnr_val = self.test(False)
            if psnr_val > max_psnr:
                max_psnr = psnr_val
        
        # torch.save(self.model.state_dict(), os.path.join(
        #          self.weightpath,'model_{}.pth'.format(self.dataname)))


    def solve_initial_frame(self):
        self.set_onlybase()
        logger.info("Training Initial Frame (Base Only)")
        self.train(epoch_n=self.init_epochs_base)
        self.remove_onlybase()
        logger.info("Training Initial Frame (SH Included)")
        self.train(epoch_n=self.init_epochs_full)
        logger.info("Point Cloud Refinement Phase Started")
        for i in range(self.args.refine_n):
            trainer.model.remove_out()
            for i in range(self.args.double_steps):
                trainer.model.repeat_pts()
            trainer.set_optimizer(self.args.lr3, self.args.lr2)
            trainer.train(epoch_n=self.init_epochs_full)

    def framedata_update(self, dset_memitem):
        self.model.framedata_update(dset_memitem)

        self.imagesgt = torch.tensor(self.model.imagesgt).float().to(device)
        self.masks = torch.tensor(self.model.masks).float().to(device)
        self.i_split = self.model.i_split
        self.imagesgt_train = self.imagesgt

    def solve_frame(self):
        prev_point_count = len(self.model.vertsparam)
        self.model.apply_narrow_band()
        inter_point_count = len(self.model.inactive_verts) + len(self.model.active_origin_verts)
        logger.info(f"Frame {self.model.frame_id} Narrow Band: Previous Point Count = {prev_point_count}, Intermediate Point Count = {inter_point_count}")
        logger.info(f"Point Breakdown: Inactive = {len(self.model.inactive_verts)}, Active Origin = {len(self.model.active_origin_verts)}")
        
        self.set_f2f_train()
        self.train(6)

        self.model.prune_and_record()
        # self.model.remove_out()
        new_point_count = len(self.model.inactive_verts) + len(self.model.active_origin_verts)
        logger.info(f"Frame {self.model.frame_id} Pruning: Intermediate Point Count = {inter_point_count}, New Point Count = {new_point_count}")
        logger.info(f"Point Breakdown: Inactive = {len(self.model.inactive_verts)}, Active Origin = {len(self.model.active_origin_verts)}")

        self.train(10)

        self.model.finalize_verts_sh()

class DataManager():
    def __init__(self, args, frame_idx_queue, dset_queue):
        self.args = args
        self.frame_idx_queue = frame_idx_queue
        self.dset_queue = dset_queue
    
    def pre_fetch_datasets(self):
        while True:
            try:
                logger.info(f'Waiting for new frame... (Queue Size: {self.frame_idx_queue.qsize()})')
                frame_idx = self.frame_idx_queue.get(block=True)
                logger.info(f'Frame received. (Queue Size: {self.frame_idx_queue.qsize()})')
            except Empty:
                logger.info('Ending data prefetch process.')
                return 
            
            logger.info(f"Finished loading frame: {frame_idx}")
            self.args.frame_id = frame_idx
            dataset = Data(self.args)
            memitem = dataset.initpc()
            self.dset_queue.put(memitem)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.freeze_support()

    args = config_args(extra_args)
    args = args.parse_args()
    device = "cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"


    summary_writer = SummaryWriter(args.train_dir)

    torch.manual_seed(20200823)
    np.random.seed(20200823)

    if args.init_model is not None:
        base_model = torch.load(args.init_model).to(device=device)
        base_model.dialation_thresh = args.dialation_thresh
        base_model.origin_vert_thresh = args.origin_vert_thresh
        base_model.origin_sh_thresh = args.origin_sh_thresh
        base_model.origin_sh_thresh_zero = args.origin_sh_thresh_zero
        base_model.disable_origin_movement = args.disable_origin_movement
        base_model.disable_origin_sh = args.disable_origin_sh

        base_model.viewdir = base_model.viewdir.to(device)
        base_model.iter_tags = torch.full((len(base_model.vertsparam),), args.frame_start).to(device=device)

        trainer = Trainer(args, base_model, device)
    else:
        args.frame_id = args.frame_start
        dataset = Data(args)
        base_memitem = dataset.initpc()
        base_model = PointSHNeRF(args, base_memitem)
        base_model = base_model.to(device=device)
        base_model.f2f_training = False
        trainer = Trainer(args, base_model, device)
        trainer.solve_initial_frame()
        if args.save_init_model:
            torch.save(base_model, f"{args.name}_base_model.pt")

    import pdb; pdb.set_trace()
    base_model.f2f_training = False
    logger.info(f"Frame {args.frame_start} Validation Accuracy")
    trainer.test(True)
    base_model.f2f_training = True

    frame_idx_queue = Queue()
    dset_queue = Queue()
    data_manager = DataManager(args, frame_idx_queue, dset_queue)
    pre_fetch_process = Process(target=data_manager.pre_fetch_datasets)
    pre_fetch_process.start()
    prefetch_factor = 3
    for i in range(1, prefetch_factor):
        frame_idx_queue.put(i + args.frame_start)

    train_frame_num = 0
    frames = []
    psnr_list = []

    for frame_idx in range(args.frame_start + 1, args.frame_end) :
        dset_memitem = dset_queue.get(block=True)
        if frame_idx + prefetch_factor < args.frame_end:
            frame_idx_queue.put(frame_idx + prefetch_factor)

        # trainer.framedata_update(dset_memitem)
        # base_model.f2f_training = False
        # trainer.remove_onlybase()
        # logger.info("Training Initial Frame (SH Included)")
        # trainer.train(epoch_n=25)

        # Update our model to work with the new frame data
        trainer.framedata_update(dset_memitem)
        logger.info(f"Solving frame: {frame_idx}")
        trainer.solve_frame()
        trainer.model.f2f_training = False
        test_psnr = trainer.test(True)
        trainer.model.f2f_training = True

        psnr_list.append(test_psnr)
        train_frame_num += 1

    logger.critical(f'average psnr {sum(psnr_list)/len(psnr_list):.4f}')
    for i in range(len(psnr_list)):
        logger.critical(f'Frame {i} psnr: {psnr_list[i]}')


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
