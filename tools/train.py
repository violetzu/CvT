# -*- coding: utf-8 -*-
# 這支程式碼用於影像分類的訓練與驗證流程。
# 主要包括以下步驟：
#   1. 解析指令引數（parse_args）。
#   2. 初始化分散式訓練（init_distributed）。
#   3. 設定 CUDNN（setup_cudnn）。
#   4. 讀取並更新配置檔（update_config），並將最終配置儲存至檔案（save_config）。
#   5. 建立模型（build_model），並使用 torchinfo.summary() 簡要顯示模型結構。
#   6. 若啟用 AMP，則可採用記憶體格式轉換為 NHWC。
#   7. 建立記錄物件（SummaryWriter）用於 TensorBoard 紀錄。
#   8. 建立優化器（build_optimizer）與恢復檢查點（resume_checkpoint）。
#   9. 構建資料載入器（build_dataloader）並進行訓練和驗證迴圈。
#  10. 每個 epoch 結束後保存檢查點，若得到最佳表現則另存為 best 模型。
#  11. 訓練完成後最終另存 final_state.pth。

from __future__ import absolute_import  # 為了確保行為與 Python 3 相容
from __future__ import division         # 在 Python 2 中啟用真實除法
from __future__ import print_function   # 在 Python 2 中使用 Python 3 的 print 函式

import argparse  # 用於解析命令列引數
import logging   # 記錄日誌
import os        # 用於檔案路徑相關作業
import pprint    # 讓輸出更易讀的 pprint
import time      # 計時相關

import torch                     # PyTorch 核心庫
import torch.nn.parallel         # 允許資料、模型平行計算
import torch.optim               # 最常用的優化器工具
from torch.utils.collect_env import get_pretty_env_info  # 獲取 PyTorch 環境資訊
from tensorboardX import SummaryWriter                  # TensorBoard 紀錄工具
from torchinfo import summary                           # 簡要顯示 PyTorch 模型結構

import _init_paths  # 自定義路徑初始化，確保可以匯入其他模組
from config import config          # 讀取預先定義的配置物件
from config import update_config   # 用於更新配置（根據外部引數）
from config import save_config     # 用於將最終配置儲存成檔案
from core.loss import build_criterion      # 建立損失函式
from core.function import train_one_epoch, test  # 訓練一個 epoch、驗證函式
from dataset import build_dataloader     # 建立資料載入器
from models import build_model           # 建立模型
from optim import build_optimizer       # 建立優化器
from scheduler import build_lr_scheduler # 建立學習率調度器
from utils.comm import comm              # 分散式訓練通訊工具
from utils.utils import create_logger    # 建立日誌器
from utils.utils import init_distributed # 初始化分散式訓練
from utils.utils import setup_cudnn      # 設定 CUDNN
from utils.utils import summary_model_on_master  # 在主進程上輸出模型摘要
from utils.utils import resume_checkpoint        # 從檢查點恢復
from utils.utils import save_checkpoint_on_master# 儲存檢查點檔案
from utils.utils import save_model_on_master     # 儲存模型檔案


def parse_args():
    # 解析使用者傳入的命令列引數
    parser = argparse.ArgumentParser(
        description='Train classification network')  # 描述這個訓練程式的用途

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)  # 指定配置檔

    # 分散式訓練相關引數
    parser.add_argument("--local_rank", type=int, default=0)  # 當前進程使用的 GPU ID
    parser.add_argument("--port", type=int, default=9000)     # 通訊埠

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)  # 其餘參數用於覆蓋設定檔內的默認值

    args = parser.parse_args()

    return args  # 回傳解析後的引數


def main():
    # 主函式，負責整個訓練與驗證流程
    args = parse_args()  # 解析傳入引數

    init_distributed(args)  # 初始化分散式訓練環境
    setup_cudnn(config)     # 設定 CUDNN（如是否使用 benchmark 等）

    update_config(config, args)  # 更新配置
    final_output_dir = create_logger(config, args.cfg, 'train')  # 建立日誌器並返回最終輸出目錄
    tb_log_dir = final_output_dir  # TensorBoard 紀錄檔的目錄

    # 只在主進程輸出環境資訊與配置
    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())  # 收集並打印 PyTorch 環境資訊
        logging.info(pprint.pformat(args))  # 打印引數
        logging.info(config)                # 打印最終的配置內容
        logging.info("=> using {} GPUs".format(args.num_gpus))  # 使用的 GPU 數量

        output_config_path = os.path.join(final_output_dir, 'config.yaml')  # 配置檔案輸出路徑
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)  # 將最終的 config 存成 YAML

    model = build_model(config)             # 依據配置建立模型
    model.to(torch.device('cuda'))          # 將模型加載到 GPU
    # summary(model, input_size=(64, 3, 224, 224))  # 顯示模型摘要

    # 在主進程上輸出模型摘要，並將模型檔案另存（若需要）
    summary_model_on_master(model, config, final_output_dir, True)

    # 若啟用 AMP 且記憶體格式為 nhwc，則將模型轉換為 channels_last
    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)

    # 建立 TensorBoardX 的 writer
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    best_perf = 0.0    # 紀錄最佳表現
    best_model = True  # 判斷目前是否為最佳模型
    begin_epoch = config.TRAIN.BEGIN_EPOCH  # 開始訓練的 epoch
    optimizer = build_optimizer(config, model)  # 建立優化器

    # 從檢查點載入（若存在），同時更新 best_perf 與 begin_epoch
    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, config, final_output_dir, True
    )

    # 建立訓練及驗證資料載入器
    train_loader = build_dataloader(config, True, args.distributed)
    valid_loader = build_dataloader(config, False, args.distributed)

    # 若為分散式訓練，包裝模型成 DistributedDataParallel
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    # 建立損失函式（訓練與評估可略有不同）
    criterion = build_criterion(config)
    criterion.cuda()
    criterion_eval = build_criterion(config, train=False)
    criterion_eval.cuda()

    # 建立學習率調度器，傳入目前 epoch 方便繼續訓練
    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    # 建立 AMP 的梯度縮放器（若啟用 AMP）
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')  # 開始訓練
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)  # 顯示當前 epoch
        logging.info('=> {} epoch start'.format(head))

        start = time.time()  # 記錄 epoch 開始時間
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)  # 分散式訓練時為了保證隨機性的一致性，需要每個 epoch 設定相同種子

        # 訓練單個 epoch
        logging.info('=> {} train start'.format(head))
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir, writer_dict,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # 在驗證集上進行評估
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(
                config, valid_loader, model, criterion_eval,
                final_output_dir, tb_log_dir, writer_dict,
                args.distributed
            )

            best_model = (perf > best_perf)  # 若本次表現優於此前最佳表現
            best_perf = perf if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )

        # 更新學習率調度器
        lr_scheduler.step(epoch=epoch+1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        # 儲存檢查點
        save_checkpoint_on_master(
            model=model,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf=best_perf,
        )

        # 若為最佳模型，則在主進程上另存為 model_best.pth
        if best_model and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, 'model_best.pth'
            )

        # 若需要保存所有 epoch 模型
        if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    # 訓練結束後，儲存 final_state.pth
    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    # 若有使用 SWA，可在此另存 swa_state.pth
    # if config.SWA.ENABLED and comm.is_main_process():
    #     save_model_on_master(
    #          args.distributed, final_output_dir, 'swa_state.pth'
    #     )

    writer_dict['writer'].close()  # 關閉 TensorBoard writer
    logging.info('=> finish training')  # 訓練流程完成


if __name__ == '__main__':
    main()
