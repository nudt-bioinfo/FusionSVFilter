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

    save_path = data_dir + 'check_bam_file_position/' + chromosome
    ut.mymkdir(save_path)
    torch.save(ins_position, save_path + '/insert' + '.pt')
    torch.save(del_position, save_path + '/delete' + '.pt')
    torch.save(n_position, save_path + '/negative' + '.pt')
    return len(ins_position),len(del_position),len(n_position)


def check_bam_file(sam_file, chromosome, begin, end, zoom):
    non_empty = 0
    read_lines = 0

    # 使用 fetch() 直接遍历返回的 reads
    for read in sam_file.fetch(chromosome, begin, end):
        read_lines += 1

    # 如果读取到的行数大于 0，则认为文件非空
    if read_lines > 0:
        non_empty = 1

    return non_empty, read_lines

def sort_and_remove_third_element(arr):
    # 按照第三个元素从大到小排序
    arr_sorted = sorted(arr, key=lambda x: x[2], reverse=True)

    # 删除每个子数组的第三个元素，只保留前两个元素
    result = [x[:2] for x in arr_sorted]

    return result

def create_image(sum_data, data_dir, bam_path, ins_len, del_len, n_len):
    chromosome, chr_len = sum_data

    print("deal chromosome " + chromosome)

    ins_position = torch.load(data_dir + 'check_bam_file_position/' + chromosome + '/insert' + '.pt')
    del_position = torch.load(data_dir + 'check_bam_file_position/' + chromosome + '/delete' + '.pt')
    n_position = torch.load(data_dir + 'check_bam_file_position/' + chromosome + '/negative' + '.pt')
    ins_len = len(ins_position)
    del_len = len(del_position)
    n_len = len(n_position)

    print(f"ins_rows:{len(ins_position)} del_rows:{len(del_position)} n_rows:{len(n_position)}")

    ins_info = []
    del_info = []
    n_info = []
    ins_non_empty_sum = 0
    del_non_empty_sum = 0
    n_non_empty_sum = 0
    sam_file = pysam.AlignmentFile(bam_path, "rb")

    ins_position_add_num = []
    del_position_add_num = []
    n_position_add_num = []
    for i, b_e in enumerate(ins_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                non_empty, read_lines = check_bam_file(sam_file, chromosome, b_e[0], b_e[1], zoom)
                ins_non_empty_sum += non_empty
                ins_position_add_num.append([b_e[0],b_e[1],read_lines])
                if non_empty == 1:
                    ins_info.append(read_lines)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)

    for i, b_e in enumerate(del_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                non_empty, read_lines = check_bam_file(sam_file, chromosome, b_e[0], b_e[1], zoom)
                del_non_empty_sum += non_empty
                del_position_add_num.append([b_e[0], b_e[1], read_lines])
                if non_empty == 1:
                    del_info.append(read_lines)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)

    for i, b_e in enumerate(n_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                non_empty, read_lines = check_bam_file(sam_file, chromosome, b_e[0], b_e[1], zoom)
                n_non_empty_sum += non_empty
                n_position_add_num.append([b_e[0], b_e[1], read_lines])
                if non_empty == 1:
                    n_info.append(read_lines)
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
    sam_file.close()

    ins_dict = {}
    del_dict = {}
    n_dict = {}
    ins_dict['read_lines'] = ins_info
    ins_dict['non_empty_num'] = ins_non_empty_sum
    ins_dict['begin_end_pair_num'] = ins_len

    del_dict['read_lines'] = del_info
    del_dict['non_empty_num'] = del_non_empty_sum
    del_dict['begin_end_pair_num'] = del_len

    n_dict['read_lines'] = n_info
    n_dict['non_empty_num'] = n_non_empty_sum
    n_dict['begin_end_pair_num'] = n_len

    ins_position_sort=sort_and_remove_third_element(ins_position_add_num)
    del_position_sort=sort_and_remove_third_element(del_position_add_num)
    n_position_sort=sort_and_remove_third_element(n_position_add_num)

    # .h5文件
    h5_save_path = data_dir + 'h5_position_sort/' + chromosome
    ut.mymkdir(h5_save_path)  # 创建文件夹
    # 创建并打开h5文件
    with h5py.File(h5_save_path + '/insert' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(ins_position_sort))
    with h5py.File(h5_save_path + '/delete' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(del_position_sort))
    with h5py.File(h5_save_path + '/negative' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(n_position_sort))

    return ins_dict, del_dict, n_dict


def save_to_json(data_dir, ins_all_dict, del_all_dict, n_all_dict):
    ins_file_name = data_dir + "ins_all_dict.json"
    del_file_name = data_dir + "del_all_dict.json"
    n_file_name = data_dir + "n_all_dict.json"

    with open(ins_file_name, 'w', encoding='utf-8') as f:
        json.dump(ins_all_dict, f, ensure_ascii=False, indent=4)


    with open(del_file_name, 'w', encoding='utf-8') as f:
        json.dump(del_all_dict, f, ensure_ascii=False, indent=4)

    with open(n_file_name, 'w', encoding='utf-8') as f:
        json.dump(n_all_dict, f, ensure_ascii=False, indent=4)


def load_from_json(data_dir):
    # 构建文件名
    ins_file_name = data_dir + "ins_all_dict.json"
    del_file_name = data_dir + "del_all_dict.json"
    n_file_name = data_dir + "n_all_dict.json"

    # 从 ins_all_dict.json 文件读取数据
    with open(ins_file_name, 'r', encoding='utf-8') as f:
        ins_all_dict = json.load(f)

    # 从 del_all_dict.json 文件读取数据
    with open(del_file_name, 'r', encoding='utf-8') as f:
        del_all_dict = json.load(f)

    # 从 n_all_dict.json 文件读取数据
    with open(n_file_name, 'r', encoding='utf-8') as f:
        n_all_dict = json.load(f)

    # 返回读取的字典
    return ins_all_dict, del_all_dict, n_all_dict


def chromosome_processing(bam_path, data_dir, ins_vcf_filename, del_vcf_filename, data_list):
    ins_all_dict = {}
    del_all_dict = {}
    n_all_dict = {}
    for i, chromosome_data in enumerate(data_list):
        ins_len, del_len, n_len = position(chromosome_data, data_dir, ins_vcf_filename, del_vcf_filename)
        ins_dict, del_dict, n_dict = create_image(chromosome_data, data_dir, bam_path, ins_len, del_len, n_len)
        # ins_all_dict[chromosome_data[0]] = ins_dict
        # del_all_dict[chromosome_data[0]] = del_dict
        # n_all_dict[chromosome_data[0]] = n_dict
    # check_bam_file_data_dir="../check_bam_file_data/"
    # save_to_json(check_bam_file_data_dir, ins_all_dict, del_all_dict, n_all_dict)


def main():
    # bam数据集路径修改
    # 金标数据集
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # 金标数据集
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR
    # bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
    # ONT
    bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

    vcf_data_dir = '../data/'
    data_dir = '../data/'

    ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
    del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

    data_list = Reference_chromosome_processing(bam_path)
    chromosome_processing(bam_path, data_dir, ins_vcf_filename, del_vcf_filename, data_list)


if __name__ == '__main__':
    main()
