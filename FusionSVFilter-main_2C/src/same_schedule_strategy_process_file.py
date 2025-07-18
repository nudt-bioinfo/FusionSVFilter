import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import utilities as ut

import threading

import struct
import json
import h5py
from cpp_module1 import chr_module
import time
import os
import multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything(2022)

# 金标数据集
# bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
# CLR
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
# ONT
bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

vcf_data_dir = "../data/"
data_dir = "../data/"

# 金标数据集
# bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
# CLR
# bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
# ONT
bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"
# NA12878
# bam_path = data_dir + "NA12878_S1.bam"
# bam_path = data_dir + "NA12878_NGMLR_sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

hight = 224
resize = torchvision.transforms.Resize([hight, hight])


def position(sum_data):
    chromosome, chr_len = sum_data
    # print(f"chromosome:{chromosome},chr_len:{chr_len}")
    ins_position = []
    del_position = []
    n_position = []
    # insert
    insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", index_col=0)
    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])

    set_pos = set()

    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))

    for pos in row_pos:
        gap = 1024
        # positive
        begin = pos - 1 - gap
        end = pos - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        ins_position.append([begin, end])

    # delete
    delete_result_data = pd.read_csv(del_vcf_filename, sep="\t", index_col=0)
    delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
    row_pos = []
    row_end = []
    for index, row in delete_chromosome.iterrows():
        row_pos.append(row["POS"])
        row_end.append(row["END"])

    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))

    for pos, end in zip(row_pos, row_end):
        gap = int((end - pos) / 4)
        if gap == 0:
            gap = 1
        # positive
        begin = pos - 1 - gap
        end = end - 1 + gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        del_position.append([begin, end])

        # negative
        del_length = end - begin

        for _ in range(2):
            end = begin

            while end - begin < del_length / 2 + 1:
                random_begin = random.randint(1, chr_len)
                while random_begin in set_pos:
                    random_begin = random.randint(1, chr_len)
                begin = random_begin - 1 - gap
                end = begin + del_length
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

            n_position.append([begin, end])

    # save_path = data_dir + 'position/' + chromosome
    # ut.mymkdir(save_path)
    # torch.save(ins_position, save_path + '/insert' + '.pt')
    # torch.save(del_position, save_path + '/delete' + '.pt')
    # torch.save(n_position, save_path + '/negative' + '.pt')

    # .h5文件
    h5_save_path = data_dir + 'h5_position/' + chromosome
    ut.mymkdir(h5_save_path)  # 创建文件夹
    # 创建并打开h5文件
    with h5py.File(h5_save_path + '/insert' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(ins_position))
    with h5py.File(h5_save_path + '/delete' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(del_position))
    with h5py.File(h5_save_path + '/negative' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(n_position))
    return len(ins_position), len(del_position), len(n_position)


def create_image(sum_data,ins_len,del_len,n_len):
    chromosome, chr_len = sum_data
    print(f"deal chromosome:{chromosome}")

    # run_threads函数参数
    data_absolute_path = "../data/"
    ins_position_path = data_absolute_path + 'h5_position/' + chromosome + '/insert' + '.h5'
    del_position_path = data_absolute_path + 'h5_position/' + chromosome + '/delete' + '.h5'
    n_position_path = data_absolute_path + 'h5_position/' + chromosome + '/negative' + '.h5'

    task_num = ins_len + del_len + n_len

    save_path = data_dir + 'image/' + chromosome
    if os.path.exists(save_path + '/negative_cigar_new_img' + '.pt'):
        return
    ut.mymkdir(save_path)

    h5_save_path = data_absolute_path + 'h5_image/' + chromosome
    ut.mymkdir(h5_save_path)
    ins_image_path = h5_save_path + '/ins_cigar_new_img' + '.h5'
    del_image_path = h5_save_path + '/del_cigar_new_img' + '.h5'
    n_image_path = h5_save_path + '/negative_cigar_new_img' + '.h5'

    # 获取CPU的核心数
    cpu_count = os.cpu_count()
    thread_num = 1
    # run_threads函数参数

    # 调用c++代码
    print(f"bam_path:{bam_path}")
    chr_module.run_threads(ins_position_path, del_position_path, n_position_path, ins_image_path, del_image_path,
                           n_image_path, ins_len, del_len, n_len, bam_path, chromosome, task_num, thread_num)

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--chr', help=help)
    parser.add_argument('--len', help=help)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    position_time_start = time.time()
    ins_len,del_len,n_len=position([args.chr, int(args.len)])
    position_time_end = time.time()

    create_image_time_start = time.time()
    create_image([args.chr, int(args.len)],ins_len,del_len,n_len)
    create_image_time_end = time.time()

    position_time = (position_time_end - position_time_start) * 1000
    create_image_time = (create_image_time_end - create_image_time_start) * 1000

    thread_id = threading.get_ident()
    # print(f"{thread_id},position time:{position_time}\n{thread_id},create_image time{create_image_time}")
