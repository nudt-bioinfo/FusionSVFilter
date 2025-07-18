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

import struct
import json
import h5py
from cpp_module1 import chr_module
import time
import os
import multiprocessing

def Reference_chromosome_processing(bam_path):
    # BAM 文件是压缩的二进制格式，通常用于存储序列比对数据
    # rb表示以二进制只读默认打开bam文件
    sam_file = pysam.AlignmentFile(bam_path, "rb")
    # 这一行代码提取 BAM 文件中所有参考染色体（或参考序列）的名称列表，sam_file.references 是一个列表，包含 BAM 文件中所有染色体的名字
    chr_list_sam_file = sam_file.references
    # sam_file.lengths 是一个列表，其中每个元素对应 chr_list_sam_file 中染色体的长度
    chr_length_sam_file = sam_file.lengths
    # 关闭打开的 BAM 文件
    sam_file.close()

    # 表示合法的染色体名称
    allowed_chromosomes = set(f"{i}" for i in range(1, 23)) | {"X", "Y"}

    chr_list = []  # 染色体名字列表
    chr_length = []  # 染色体的长度

    for chrom, length in zip(chr_list_sam_file, chr_length_sam_file):
        if chrom in allowed_chromosomes:
            chr_list.append(chrom)
            chr_length.append(length)

    data_list = []  # 存储(chromosome, chr_len)元组
    for chromosome, chr_len in zip(chr_list, chr_length):
        data_list.append((chromosome, chr_len))
    return data_list
    
def parse_args():
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--thread_num', help=help)
    args = parser.parse_args()
    return args

def process_chromosome(chr, length, cpu_core):
    command = f"taskset -c {cpu_core} python chromosome_check_bam_file_process.py --chr {chr} --len {length}"
    print(command)
    subprocess.Popen(command, shell=True).wait()

def worker(chromosome_data, cpu_core):
    chr, length = chromosome_data
    process_chromosome(chr, length, cpu_core)

def main():
    # bam数据集路径修改
    # 金标数据集
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # 金标数据集
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR
    bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
    # ONT
    # bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

    vcf_data_dir = '../data/'
    data_dir = '../data/'

    ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
    del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

    seed_everything(2022)

    args = parse_args()
    thread_num = int(args.thread_num)
    
    data_list = Reference_chromosome_processing(bam_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        futures = []
        for i, chromosome_data in enumerate(data_list):
            core_index = i % thread_num
            futures.append(executor.submit(worker, chromosome_data, core_index))
        concurrent.futures.wait(futures)

if __name__ == '__main__':
    main()
