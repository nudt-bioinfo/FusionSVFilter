import argparse
import os
import random
import subprocess
import concurrent.futures

import numpy as np
import pandas as pd
import pysam
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import utilities as ut
import time

start_time = time.time()


def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--thread_num', help=help)
    args = parser.parse_args()
    return args


def process_chromosome(chr, length, cpu_core):
    command = f"taskset -c {cpu_core} python same_schedule_strategy_process_file.py --chr {chr} --len {length}"
    print(command)
    subprocess.Popen(command, shell=True).wait()


seed_everything(2022)

# 金标数据集
# bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
# CLR
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
# ONT
bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

vcf_data_dir = "../data/"

# 金标数据集
# bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
# CLR
# bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
# ONT
bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"

sam_file = pysam.AlignmentFile(bam_path, "rb")
chr_list_sam_file = sam_file.references
chr_length_sam_file = sam_file.lengths
sam_file.close()

allowed_chromosomes = set(f"{i}" for i in range(1, 23)) | {"X", "Y"}

chr_list = []
chr_length = []

for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
    if chrom in allowed_chromosomes:
        chr_list.append(chrom)
        chr_length.append(length)
        # print(f"chrom:{chrom} length:{length}")

# for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
#     chr_list.append(chrom)
#     chr_length.append(length)
#     #print(f"chrom:{chrom} length:{length}")

hight = 224

data_list = []
for chromosome, chr_len in zip(chr_list, chr_length):
    data_list.append((chromosome, chr_len))

args = parse_args()
thread_num = int(args.thread_num)


def worker(chromosome_data, cpu_core):
    chr, length = chromosome_data
    process_chromosome(chr, length, cpu_core)

with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
    futures = []
    for i, chromosome_data in enumerate(data_list):
        core_index = i % thread_num
        futures.append(executor.submit(worker, chromosome_data, core_index))
    concurrent.futures.wait(futures)
end_time = time.time()
elapsed_time = (end_time - start_time) * 1000
print(f"total time:{elapsed_time:.3f}")

# def data_type_transfer(chromosome_data):
#     chromosome,chr_len=chromosome_data
#     h5_save_path="../data/"+'h5_image/' + chromosome
#     ins_h5_image_path = h5_save_path + '/ins_cigar_new_img' + '.h5'
#     del_h5_image_path = h5_save_path + '/del_cigar_new_img' + '.h5'
#     n_h5_image_path = h5_save_path + '/negative_cigar_new_img' + '.h5'
#
#     pt_save_path = "../data/" + 'image/' + chromosome
#     ins_pt_image_path=pt_save_path+'/ins_cigar_new_img'+'.pt'
#     del_pt_image_path = pt_save_path + '/del_cigar_new_img' + '.pt'
#     n_pt_image_path = pt_save_path + '/negative_cigar_new_img' + '.pt'
#
#     # 打开b.h5文件
#     with h5py.File(ins_h5_image_path, 'r') as file:
#         # 从文件中读取四维数组
#         ins_img_data = file['array'][:]
#         ins_cigar_img = torch.tensor(ins_img_data.astype(np.float32))
#
#     with h5py.File(del_h5_image_path, 'r') as file:
#         # 从文件中读取四维数组
#         del_img_data = file['array'][:]
#         del_cigar_img = torch.tensor(del_img_data.astype(np.float32))#属于计算密集型
#
#     with h5py.File(n_h5_image_path, 'r') as file:
#         # 从文件中读取四维数组
#         n_img_data = file['array'][:]
#         negative_cigar_img = torch.tensor(n_img_data.astype(np.float32))
#
#     torch.save(ins_cigar_img, ins_pt_image_path)
#     torch.save(del_cigar_img, del_pt_image_path)
#     torch.save(negative_cigar_img, n_pt_image_path)
#     print(f"染色体：{chromosome}的数据类型转换完成。")
#
# # 串行
# data_transfer_time_start = time.time()
# for i, chromosome_data in enumerate(data_list):
#     data_type_transfer(chromosome_data)
# data_transfer_time_end = time.time()
# elapsed_data_transfer_time = (data_transfer_time_end - data_transfer_time_start) * 1000
# print(f"数据类型转换时间：{elapsed_data_transfer_time:.3f}ms")
