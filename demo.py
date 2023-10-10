# -*- coding:utf-8 -*-
# @Time       :2023/8/18 10:35 AM
# @AUTHOR     :Duo Wang
# @FileName   :demo.py
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import time
import os

import argparse
import logging
from models.MHST_Net import MHST
from utils import (trPixel2Patch, tsPixel2Patch, set_seed,
                   output_metric, train_epoch, valid_epoch, draw_classification_map)

# -------------------------------------------------------------------------------
# create log
logger = logging.getLogger("Trainlog")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
file_handler = logging.FileHandler("cls_logs/test_Houston.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)
# -------------------------------------------------------------------------------
# Parameter Setting
parser = argparse.ArgumentParser(description="Training for MHST-Net")
parser.add_argument('--gpu_id', default='0',
                    help='gpu id')
parser.add_argument('--seed', type=int, default=0,
                    help='number of seed')
parser.add_argument('--batch_size', type=int, default=64,
                    help='number of batch size')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epoch')
parser.add_argument('--dataset', choices=['Trento', 'Houston'], default='Houston',
                    help='dataset to use')
parser.add_argument('--num_classes', choices=[6, 15], default=15,
                    help='number of classes')
parser.add_argument('--test_freq', type=int, choices=[5, 8], default=8,
                    help='number of evaluation for Trento & Houston respectively')
parser.add_argument('--learning_rate', type=float, choices=[0.0005, 0.0008], default=0.0008,
                    help='learning rate for Trento & Houston respectively')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='')
parser.add_argument('--step_size', type=int, default=100,
                    help='')
parser.add_argument('--coefficient_hsi', type=float, default=0.6,
                    help='weight of HSI data in feature fusion, LiDAR:(1-coefficient_hsi)')
parser.add_argument('--coefficient_vit', type=float, default=0.7,
                    help='weight of ViT cls result in fusion classification, CNN:(1-coefficient_vit)')
parser.add_argument('--flag', choices=['train', 'test'], default='test',
                    help='testing mark')

parser.add_argument('--patch_size', type=int, default=8,
                    help='cnn input size')
parser.add_argument('--en_depth', type=int, default=5,
                    help='depth of vit encoder')
parser.add_argument('--en_heads', type=int, default=4,
                    help='number of heads in vit attn')
parser.add_argument('--encoder_embed_dim', type=int, default=64,
                    help='number of channels in vit input data')
parser.add_argument('--num_patches', type=int, default=64,
                    help='number of patch')
parser.add_argument('--hsp_vit_depth', type=int, default=8,
                    help='depth of head selection vit')
parser.add_argument('--hsp_vit_num_heads', type=int, default=16,
                    help='number of heads in head selection vit attn')
parser.add_argument('--use_head_select', type=bool, default=True,
                    help='if True, use head_select')
parser.add_argument('--hsp_qkv_bias', type=bool, default=False,
                    help='if True, use qkv_bias in head selection vit attn')
parser.add_argument('--head_tau', type=int, default=5,
                    help='param of head selection vit')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)


# -------------------------------------------------------------------------------
# train & val
def train1time():
    # -------------------------------------------------------------------------------
    if args.dataset == 'Houston':
        DataPath1 = 'Data/Houston/Houston.mat'
        DataPath2 = 'Data/Houston/LiDAR.mat'
        LabelPath = 'Data/Houston/train_test_gt.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
    elif args.dataset == 'Trento':
        DataPath1 = 'Data/Trento/HSI.mat'
        DataPath2 = 'Data/Trento/LiDAR.mat'
        Data1 = loadmat(DataPath1)['HSI']
        Data2 = loadmat(DataPath2)['LiDAR']
        LabelPath = 'Data/Trento/train_test_gt.mat'
    else:
        raise "Correct dataset needed!"

    Data1 = Data1.astype(np.float32)  # hsi
    Data2 = Data2.astype(np.float32)  # lidar
    TrLabel = loadmat(LabelPath)['train_data']
    TsLabel = loadmat(LabelPath)['test_data']

    patchsize = args.patch_size  # input spatial size for CNN
    pad_width = np.floor(patchsize / 2)
    pad_width = int(pad_width)
    TrainPatch1, TrainPatch2, TrainLabel = trPixel2Patch(
        Data1, Data2, patchsize, pad_width, TrLabel)
    TestPatch1, TestPatch2, TestLabel, _, _ = tsPixel2Patch(
        Data1, Data2, patchsize, pad_width, TsLabel)

    train_dataset = Data.TensorDataset(
        TrainPatch1, TrainPatch2, TrainLabel)
    test_dataset = Data.TensorDataset(
        TestPatch1, TestPatch2, TestLabel)
    train_loader = Data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)

    [H1, W1, _] = np.shape(Data1)
    Data2 = Data2.reshape([H1, W1, -1])
    height1, width1, band1 = Data1.shape
    height2, width2, band2 = Data2.shape
    # data size
    print('\n')
    logger.info("=" * 50)
    logger.info("=" * 50)
    logger.info("hsi_height={0},hsi_width={1},hsi_band={2}".format(height1, width1, band1))
    logger.info("lidar_height={0},lidar_width={1},lidar_band={2}".format(height2, width2, band2))
    # -------------------------------------------------------------------------------
    # create model
    model = MHST(l1=band1, l2=band2, patch_size=args.patch_size,
                 num_patches=args.num_patches, num_classes=args.num_classes,
                 encoder_embed_dim=args.encoder_embed_dim,
                 en_depth=args.en_depth, en_heads=args.en_heads,
                 mlp_dim=8, dropout=0.1, emb_dropout=0.1,
                 coefficient_hsi=args.coefficient_hsi, coefficient_vit=args.coefficient_vit,
                 hsp_vit_depth=args.hsp_vit_depth, hsp_vit_num_heads=args.hsp_vit_num_heads,
                 head_tau=args.head_tau, use_head_select=args.use_head_select,
                 vit_qkv_bias=args.hsp_qkv_bias,
                 mlp_ratio=4, attnproj_mlp_drop=0.1, attn_drop=0.1)

    model = model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # -------------------------------------------------------------------------------
    # train & test
    if args.flag == 'train':
        BestAcc = 0
        get_ts_result = False
        val_acc = []
        logger.info("start training")
        tic = time.time()
        for epoch in range(args.epochs):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(
                model, train_loader, criterion, optimizer)
            OA1, AA1, Kappa1, CA1 = output_metric(tar_t, pre_t)
            logger.info("Epoch: {:03d} | train_loss: {:.4f} | train_OA: {:.4f} | train_AA: {:.4f} | train_Kappa: {:.4f}"
                        .format(epoch + 1, train_obj, OA1, AA1, Kappa1))
            scheduler.step()

            if ((epoch + 1) % args.test_freq == 0) | (epoch == args.epochs - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
                OA2, AA2, Kappa2, CA2 = output_metric(tar_v, pre_v)
                val_acc.append(OA2)
                logger.info("Every {} epochs' records:".format(args.test_freq))
                logger.info(
                    "OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA2, Kappa2))
                logger.info(CA2)
                if AA2 > BestAcc:
                    torch.save(model.state_dict(), 'cls_param/MHSTNet_{}.pkl'.format(args.dataset))
                    BestAcc = AA2

        toc = time.time()
        model.eval()
        model.load_state_dict(torch.load('cls_param/MHSTNet_{}.pkl'.format(args.dataset)))
        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        logger.info("Final records:")
        logger.info("Maximal Accuracy: %f, index: %i" % (max(val_acc), val_acc.index(max(val_acc))))
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Running Time: {:.2f}".format(toc - tic))
        args.flag = 'test'

    if args.flag == 'test':
        # test best model
        get_ts_result = False
        model.eval()
        model.load_state_dict(torch.load('cls_param/MHSTNet_{}.pkl'.format(args.dataset)))

        tar_v, pre_v = valid_epoch(model, test_loader, criterion, get_ts_result)
        OA, AA, Kappa, CA = output_metric(tar_v, pre_v)
        logger.info("Test records:")
        logger.info("Maximal Accuracy: %f" % OA)
        logger.info("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA, AA, Kappa))
        logger.info(CA)
        logger.info("Parameter:")
        logger.info(vars(args))

        # draw map
        if args.dataset == 'Houston':
            TR_TS_Path = 'Data/Houston/tr_ts.mat'
        elif args.dataset == 'Trento':
            TR_TS_Path = 'Data/Trento/tr_ts.mat'
        else:
            raise "Correct dataset needed!"

        TR_TS_Label = loadmat(TR_TS_Path)['tr_ts']
        # draw gt map
        draw_classification_map(TR_TS_Label, 'cls_map/{}_groundTruth.png'.format(args.dataset), args.dataset)
        # draw cls map
        TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label, xIndex_list, yIndex_list = tsPixel2Patch(
            Data1, Data2, patchsize, pad_width, TR_TS_Label)
        TR_TS_dataset = Data.TensorDataset(
            TR_TS_Patch1, TR_TS_Patch2, TR_TS_Label)
        best_test_loader = Data.DataLoader(
            TR_TS_dataset, batch_size=args.batch_size, shuffle=False)

        get_ts_result = True  # if True, return cls result
        ts_result = valid_epoch(model, best_test_loader, criterion, get_ts_result)
        ts_result_matrix = np.full((H1, W1), 0)
        for i in range(len(ts_result)):
            ts_result_matrix[xIndex_list[i], yIndex_list[i]] = ts_result[i]
        draw_classification_map(ts_result_matrix, 'cls_map/{}_predLabel.png'.format(args.dataset), args.dataset)


if __name__ == '__main__':
    set_seed(args.seed)
    train1time()
