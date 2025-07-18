import utilities as ut
import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
import os
from net import IDENet
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from multiprocessing import Pool, cpu_count
import pysam
import time
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest import Repeater
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
import list2img
from hyperopt import hp

seed_everything(2022)

#HiFi
bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
#CLR
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
#ONT
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

#HiFi
bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
#CLR
# bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
#ONT
# bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

# bam_data_dir = "../data/"
# vcf_data_dir = "../data/"
# data_dir = "../data/"
vcf_data_dir = "/home/zxh/D/program/CSV-Filter-main_2C/data/"
data_dir = "/home/zxh/D/program/CSV-Filter-main_2C/data/"
bs = 16
# my_label = "ResNet34"
my_label = "ResNet50"

# bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

# get chr list
sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list = sam_file.references
chr_length = sam_file.lengths
sam_file.close()

hight = 224

# 初始化TensorBoard日志记录器，指定日志存储路径
logger = TensorBoardLogger(os.path.join(
    data_dir, "channel_predict"), name=my_label)

#设置模型保存的回调，监控validation_mean指标，在训练过程中保存最好的模型。
checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints_predict/" + my_label,
    filename='{epoch:02d}-{validation_mean:.2f}-{train_mean:.2f}',
    monitor="validation_mean",
    verbose=False,
    save_last=None,
    save_top_k=1,
    mode="max",
    auto_insert_metric_name=True,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=None,
    save_on_train_epoch_end=None,
    every_n_val_epochs=None
)

#没有用到
def main_train():
    config = {
        "lr": 7.1873e-6,
        "batch_size": 14,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
    }

    model = IDENet(data_dir, config)

    trainer = pl.Trainer(
        max_epochs=30,
        gpus=1,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)

# 1 usage
#用于训练的函数
def train_tune(config, checkpoint_dir=None, num_epochs=200, num_gpus=1):
    model = IDENet(data_dir, config)#创建一个IDENet模型实例

    #冻结模型中包含“conv2ds”字段的卷积层参数，不参与训练。
    for name, param in model.named_parameters():
        if "conv2ds" in name:
            param.requires_grad = False

    # pl.Trainer是pytorch-lightning的训练器，负责训练过程的管理
    trainer = pl.Trainer(
        max_epochs=num_epochs,#最大训练论述
        # gpus=num_gpus,#使用gpu数量
        gpus=[0],
        check_val_every_n_epoch=1,#每隔多少个epoch进行依次验证
        logger=logger,
        callbacks=[checkpoint_callback],#使用的回调，这里是模型检查点回调
        precision=16 # 使用16位精度训练（可以加速训练并节省显存）
    )
    trainer.fit(model)
# 没有用到
class MyStopper(tune.Stopper):
    def __init__(self, metric, value, epoch=1):
        self._metric = metric
        self._value = value
        self._epoch = epoch

    def __call__(self, trial_id, result):
        """Return a boolean representing if the tuning has to stop."""
        return (result["training_iteration"] > self._epoch) and (result[self._metric] < self._value)

    def stop_all(self):
        """Return whether to stop and prevent trials from starting."""
        return False

#执行超参数调优
def gan_tune(num_samples=-1, num_epochs=50, gpus_per_trial=1):
    config = {#定义要调优的超参数配置
        "lr": tune.loguniform(1e-7, 1e-3),#学习率通过对数均匀分布进行采样，范围在1e-7到1e-3之间
        "batch_size": bs,#批次大小
        "beta1": 0.9, #用于优化器的超参数
        "beta2": 0.999, 
        'weight_decay': tune.uniform(0, 1e-4),
        'model': "resnet50" # 使用resnet作为模型
    }

    # 使用贝叶斯优化搜索算法，寻找最佳超参数。validation_mean表示优化指标，max表示目标是最大化该指标
    bayesopt = HyperOptSearch(config, metric="validation_mean", mode="max")
    re_search_alg = Repeater(bayesopt, repeat=1)#设置贝叶斯优化算法重复执行一次

    #用于调度超参数搜索过程
    scheduler = ASHAScheduler(
        max_t=num_epochs,#最大训练周期
        grace_period=1,
        reduction_factor=2,
    )

    #用于打印训练过程中的进度信息
    reporter = CLIReporter(
        metric_columns=['train_loss', "train_mean",
                        'validation_loss', "validation_mean"]
    )

    #执行超参数调优的主函数
    analysis = tune.run(
        tune.with_parameters(#包装train_tune函数，并传递参数
            train_tune, # 微调训练函数
            num_epochs=num_epochs,
        ),
        local_dir=data_dir,
        resources_per_trial={#每个实验所需的计算资源
            "cpu": 1,
            "gpu": 1,
        },
        num_samples=num_samples,#搜索样本数量
        metric='validation_mean',#优化的目标指标
        mode='max',#表示目标是最大化目标指标
        scheduler=scheduler,#使用的调度器
        progress_reporter=reporter,#报告进度
        resume=False,
        search_alg=re_search_alg,#使用贝叶斯优化算法进行搜索
        max_failures=-1,
        name="tune")

ray.init()#初始化ray集群，为分布式计算做好准备
gan_tune()#启动超参数调优的过程
