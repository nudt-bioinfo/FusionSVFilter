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


def Reference_chromosome_processing():
    # 产生随机数，用于随机数据集，以及用相同的随机数种子比较不同的深度学习模型
    seed_everything(2022)

    # bam数据集路径修改
    # 金标数据集
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR
    # bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # 金标数据集
    bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR
    # bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
    # ONT
    # bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

    vcf_data_dir = "../data/"

    vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"

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


def position(sum_data, data_dir, ins_vcf_filename, del_vcf_filename):
    chromosome, chr_len = sum_data  # 参数表示染色体编号和染色体长度
    ins_position = []  # insert
    del_position = []  # delete
    n_position = []  # negtive
    # insert
    # pd.read_csv()：这是 pandas 库中的一个函数，用于读取 CSV（逗号分隔值）格式的文件.虽然函数名是 read_csv，但它不仅限于 CSV 格式，还可以读取其他分隔符分隔的数据文件
    # sep="\t" 表明文件中的数据是以 制表符（Tab） 分隔的
    # 这个参数指定读取文件时，第一列（索引列）将作为数据框的索引列
    insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", index_col=0)
    # 这行代码用于从 insert_result_data（一个 DataFrame）中筛选出符合特定条件的行
    # insert_chromosome = ...：最后，符合该条件的所有行会被提取并存储在新的 DataFrame insert_chromosome 中
    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    # 使用 for 循环遍历 insert_chromosome 这个数据框的每一行
    # iterrows() 是 Pandas 提供的一个方法，它会返回每一行的索引 (index) 和行数据 (row)
    # 数据中有索引时使用iterrows()，数据中没有索引时可以用enumerate为每条数据生成索引
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])
    # 集合在 Python 中是一个无序且不重复的元素集合
    set_pos = set()
    # range用法：range(1,10)=1,2,...,9
    for pos in row_pos:
        # 使用 update() 方法确保 set_pos 中只包含唯一值。如果这个范围内的某些值已经存在于 set_pos 中，它们不会被重复添加
        set_pos.update(range(pos - 1024, pos + 1024))  # 在相应的insert位置+-1024，用于表示窗口大小

    # 遍历row_pos中的每个insertSV点
    for pos in row_pos:
        gap = 1024
        # positive
        begin = pos - 1 - gap
        end = pos - 1 + gap  # 总大小=2048
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
        set_pos.update(range(pos - 1024, pos + 1024))  # set_pos中存储deleteSV点中的窗口值

    for pos, end in zip(row_pos, row_end):
        gap = int((end - pos) / 4)
        if gap == 0:
            gap = 1
        # positive
        begin = pos - 1 - gap
        end = end - 1 + gap  # 总长度=6gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        del_position.append([begin, end])

        # negative
        del_length = end - begin  # 等于6gap

        for _ in range(2):
            end = begin

            while end - begin < del_length / 2 + 1:
                random_begin = random.randint(1, chr_len)
                while random_begin in set_pos:  # 如果在deleteSV点窗口内，那么重新生成随机数
                    random_begin = random.randint(1, chr_len)
                # 最终生成随机开始值一定不在deleteSV点窗口内
                begin = random_begin - 1 - gap
                end = begin + del_length  # 满足end-begin=del_length
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

            n_position.append([begin, end])
    print(f"chr:{chromosome}:len(ins):{len(ins_position)},len(del):{len(del_position)},len(n):{len(n_position)}")

    # .pt文件
    pt_save_path = data_dir + 'position/' + chromosome
    ut.mymkdir(pt_save_path)
    torch.save(ins_position, pt_save_path + '/insert' + '.pt')
    torch.save(del_position, pt_save_path + '/delete' + '.pt')
    torch.save(n_position, pt_save_path + '/negative' + '.pt')

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


def create_image(sum_data, data_dir, bam_path, ins_len, del_len, n_len):
    chromosome, chr_len = sum_data
    print(f"deal chromosome:{chromosome}")

    # run_threads函数参数
    data_absolute_path = "../data/"
    ins_position_path = data_absolute_path + 'h5_position/' + chromosome + '/insert' + '.h5'
    del_position_path = data_absolute_path + 'h5_position/' + chromosome + '/delete' + '.h5'
    n_position_path = data_absolute_path + 'h5_position/' + chromosome + '/negative' + '.h5'

    # data_absolute_path = "../data/"
    # ins_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/insert' + '.h5'
    # del_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/delete' + '.h5'
    # n_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/negative' + '.h5'

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
    thread_num = 16
    # run_threads函数参数

    # 调用c++代码
    print(f"bam_path:{bam_path}")
    chr_module.run_threads(ins_position_path, del_position_path, n_position_path, ins_image_path, del_image_path,
                           n_image_path, ins_len, del_len, n_len, bam_path, chromosome, task_num, thread_num)

all_ins=0
all_del=0

def One_chromosome_processing(chromosome_data):
    # bam数据集路径修改
    # 金标数据集
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR
    # bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # 金标数据集
    bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR
    # bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
    # ONT
    # bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

    # HiFi
    # bam_path="/home/zxh/D/program/CSV-Filter-main/data/split_bams/HiFi_bam/"+ chromosome_data[0] + ".bam"
    # CLR
    # bam_path = "/home/zxh/D/program/CSV-Filter-main/data/split_bams/CLR_bam/" + chromosome_data[0] + ".bam"
    # ONT
    # bam_path = "/home/zxh/D/program/CSV-Filter-main/data/split_bams/ONT_bam/" + chromosome_data[0] + ".bam"

    vcf_data_dir = '../data/'
    data_dir = '../data/'

    ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
    del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"
    ins_len, del_len, n_len = position(chromosome_data, data_dir, ins_vcf_filename, del_vcf_filename)
    return ins_len,del_len

data_list = Reference_chromosome_processing()
for i, chromosome_data in enumerate(data_list):
    ins_len,del_len=One_chromosome_processing(chromosome_data)
    all_ins+=ins_len
    all_del+=del_len
print(f"ins_len:{all_ins},del_len:{all_del}")

