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
    # HiFi-pbmm2
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # HiFi
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR-pbmm2
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # CLR
    # bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # HiFi-pbmm2
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-pbmm2.sorted.bam"
    # HiFi
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR-pbmm2
    bam_path = bam_data_dir + "HG002-PacBio-CLR-pbmm2.sorted.bam"
    # CLR
    # bam_path = bam_data_dir + "HG002-PacBio-CLR-minimap2.sorted.bam"
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

def position(sum_data,data_dir,ins_vcf_filename,del_vcf_filename):
    chromosome, chr_len = sum_data#参数表示染色体编号和染色体长度
    ins_position = []#insert
    del_position = []#delete
    n_position = []#negtive
    # insert
    #pd.read_csv()：这是 pandas 库中的一个函数，用于读取 CSV（逗号分隔值）格式的文件.虽然函数名是 read_csv，但它不仅限于 CSV 格式，还可以读取其他分隔符分隔的数据文件
    #sep="\t" 表明文件中的数据是以 制表符（Tab） 分隔的
    #这个参数指定读取文件时，第一列（索引列）将作为数据框的索引列
    insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", index_col=0, low_memory=False)
    #这行代码用于从 insert_result_data（一个 DataFrame）中筛选出符合特定条件的行
    #insert_chromosome = ...：最后，符合该条件的所有行会被提取并存储在新的 DataFrame insert_chromosome 中
    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    ins_row_image_name=[] # image_name
    #使用 for 循环遍历 insert_chromosome 这个数据框的每一行
    # iterrows() 是 Pandas 提供的一个方法，它会返回每一行的索引 (index) 和行数据 (row)
    #数据中有索引时使用iterrows()，数据中没有索引时可以用enumerate为每条数据生成索引
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])
        ins_row_image_name.append(row["ID"])
    #集合在 Python 中是一个无序且不重复的元素集合
    set_pos = set()
    #range用法：range(1,10)=1,2,...,9
    for pos in row_pos:
        #使用 update() 方法确保 set_pos 中只包含唯一值。如果这个范围内的某些值已经存在于 set_pos 中，它们不会被重复添加
        set_pos.update(range(pos - 1024, pos + 1024))#在相应的insert位置+-1024，用于表示窗口大小

    #遍历row_pos中的每个insertSV点
    for pos in row_pos:
        gap = 1024
        # positive
        begin = pos - 1 - gap
        end = pos - 1 + gap#总大小=2048
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        ins_position.append([begin, end])

    # delete
    delete_result_data = pd.read_csv(del_vcf_filename, sep="\t", index_col=0, low_memory=False)
    delete_chromosome = delete_result_data[delete_result_data["CHROM"] == chromosome]
    row_pos = []
    row_end = []
    del_row_image_name = []
    for index, row in delete_chromosome.iterrows():
        row_pos.append(row["POS"])
        row_end.append(row["END"])
        del_row_image_name.append(row["ID"])

    for pos in row_pos:
        set_pos.update(range(pos - 1024, pos + 1024))#set_pos中存储deleteSV点中的窗口值

    for pos, end in zip(row_pos, row_end):
        gap = int((end - pos) / 4)
        if gap == 0:
            gap = 1
        # positive
        begin = pos - 1 - gap
        end = end - 1 + gap#总长度=6gap
        if begin < 0:
            begin = 0
        if end >= chr_len:
            end = chr_len - 1

        del_position.append([begin, end])

        # negative
        del_length = end - begin#等于6gap

        for _ in range(2):
            end = begin

            while end - begin < del_length / 2 + 1:
                random_begin = random.randint(1, chr_len)
                while random_begin in set_pos:#如果在deleteSV点窗口内，那么重新生成随机数
                    random_begin = random.randint(1, chr_len)
                #最终生成随机开始值一定不在deleteSV点窗口内
                begin = random_begin - 1 - gap
                end = begin + del_length#满足end-begin=del_length
                if begin < 0:
                    begin = 0
                if end >= chr_len:
                    end = chr_len - 1

            n_position.append([begin, end])

    TEST=True
    if TEST==True:
        # n_position1=[]
        # n_position1.append(n_position[0])
        n_position=[]

    # .txt文件，存储文件名
    txt_image_name_save_path=data_dir + 'image_name/' + chromosome
    ut.mymkdir(txt_image_name_save_path)
    with open(txt_image_name_save_path + '/insert' + '.txt',"w") as txt_file:
        for item in ins_row_image_name:
            txt_file.write(item + "\n")
    with open(txt_image_name_save_path + '/delete' + '.txt',"w") as txt_file:
        for item in del_row_image_name:
            txt_file.write(item + "\n")

    # .pt文件
    pt_save_path = data_dir + 'position/' + chromosome
    ut.mymkdir(pt_save_path)
    torch.save(ins_position, pt_save_path + '/insert' + '.pt')
    torch.save(del_position, pt_save_path + '/delete' + '.pt')
    torch.save(n_position, pt_save_path + '/negative' + '.pt')

    #.h5文件
    h5_save_path = data_dir + 'h5_position/' + chromosome
    ut.mymkdir(h5_save_path)  # 创建文件夹
    # 创建并打开h5文件
    with h5py.File(h5_save_path+'/insert'+'.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(ins_position))
    with h5py.File(h5_save_path+'/delete'+'.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(del_position))
    with h5py.File(h5_save_path+'/negative'+'.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(n_position))
    return len(ins_position),len(del_position),len(n_position)

def create_image(sum_data,data_dir,bam_path,ins_len,del_len,n_len):
    chromosome, chr_len = sum_data
    print(f"deal chromosome:{chromosome}")

    # run_threads函数参数
    data_absolute_path=data_dir
    ins_position_path=data_absolute_path + 'h5_position/' + chromosome + '/insert' + '.h5'
    del_position_path = data_absolute_path + 'h5_position/' + chromosome + '/delete' + '.h5'
    n_position_path = data_absolute_path + 'h5_position/' + chromosome + '/negative' + '.h5'

    # data_absolute_path = "../data/"
    # ins_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/insert' + '.h5'
    # del_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/delete' + '.h5'
    # n_position_path = data_absolute_path + 'h5_position_sort/' + chromosome + '/negative' + '.h5'

    task_num=ins_len+del_len+n_len

    save_path = data_dir + 'image/' + chromosome
    if os.path.exists(save_path + '/negative_cigar_new_img' + '.pt'):
        return
    ut.mymkdir(save_path)

    h5_save_path=data_absolute_path + 'h5_image/' + chromosome
    ut.mymkdir(h5_save_path)
    ins_image_path = h5_save_path + '/ins_cigar_new_img' + '.h5'
    del_image_path = h5_save_path + '/del_cigar_new_img' + '.h5'
    n_image_path = h5_save_path + '/negative_cigar_new_img' + '.h5'

    # 获取CPU的核心数
    cpu_count = os.cpu_count()
    thread_num=16
    # run_threads函数参数

    #调用c++代码
    print(f"bam_path:{bam_path}")
    chr_module.run_threads(ins_position_path,del_position_path,n_position_path,ins_image_path,del_image_path,n_image_path,ins_len,del_len,n_len,bam_path,chromosome,task_num,thread_num)

def One_chromosome_processing(chromosome_data):
    #bam数据集路径修改
    # HiFi-pbmm2
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # HiFi
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
    # CLR-pbmm2
    bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # CLR
    # bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
    # ONT
    # bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

    # HiFi-pbmm2
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-pbmm2.sorted.bam"
    # HiFi
    # bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
    # CLR-pbmm2
    bam_path = bam_data_dir + "HG002-PacBio-CLR-pbmm2.sorted.bam"
    # CLR
    # bam_path = bam_data_dir + "HG002-PacBio-CLR-minimap2.sorted.bam"
    # ONT
    # bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"

    #HiFi
    # bam_path="/home/zxh/D/program/CSV-Filter-main/data/split_bams/HiFi_bam/"+ chromosome_data[0] + ".bam"
    #CLR
    # bam_path = "/home/zxh/D/program/CSV-Filter-main/data/split_bams/CLR_bam/" + chromosome_data[0] + ".bam"
    #ONT
    # bam_path = "/home/zxh/D/program/CSV-Filter-main/data/split_bams/ONT_bam/" + chromosome_data[0] + ".bam"

    vcf_data_dir = '../data/'
    data_dir = '../data/'
    
    ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
    del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"
    
    #position时间测量
    position_time_start = time.time()
    ins_len,del_len,n_len=position(chromosome_data,data_dir,ins_vcf_filename,del_vcf_filename)
    position_time_end = time.time()
    
    #create_image时间测量
    create_image_time_start = time.time()
    create_image(chromosome_data,data_dir,bam_path,ins_len,del_len,n_len)
    create_image_time_end = time.time()

    position_time = (position_time_end - position_time_start) * 1000
    create_image_time = (create_image_time_end - create_image_time_start) * 1000
    print(f"position time:{position_time}ms\ncreate_image time:{create_image_time}ms")

def data_type_transfer_multithreads(chromosome_data,cpu_core):
    chromosome,chr_len=chromosome_data
    # 使用 taskset 工具来指定某个 CPU 核心运行 Python 脚本 image_encoding_file.py，并将染色体编号和长度作为参数传递给它
    command = f"taskset -c {cpu_core} python data_from_h5_to_pt.py --chr {chromosome} --len {chr_len}"
    print(command)
    # subprocess.Popen(command, shell=True) 会创建一个新的进程来运行上面构建的命令
    # wait() 方法会使当前进程等待，直到被调用的子进程完成。这意味着该函数会阻塞，直到 image_encoding_file.py 完成执行
    subprocess.Popen(command, shell=True).wait()

def worker_thread(chromosome_data,core_index):
    # 在子进程中执行 data_type_transfer
    data_type_transfer_multithreads(chromosome_data,core_index)

def data_type_transfer(chromosome_data):
    chromosome,chr_len=chromosome_data
    h5_save_path="../data/"+'h5_image/' + chromosome
    ins_h5_image_path = h5_save_path + '/ins_cigar_new_img' + '.h5'
    del_h5_image_path = h5_save_path + '/del_cigar_new_img' + '.h5'
    n_h5_image_path = h5_save_path + '/negative_cigar_new_img' + '.h5'

    pt_save_path = "../data/" + 'image/' + chromosome
    ins_pt_image_path=pt_save_path+'/ins_cigar_new_img'+'.pt'
    del_pt_image_path = pt_save_path + '/del_cigar_new_img' + '.pt'
    n_pt_image_path = pt_save_path + '/negative_cigar_new_img' + '.pt'

    # 打开b.h5文件
    with h5py.File(ins_h5_image_path, 'r') as file:
        # 从文件中读取四维数组
        ins_img_data = file['array'][:]
        ins_cigar_img = torch.tensor(ins_img_data.astype(np.float32))

    with h5py.File(del_h5_image_path, 'r') as file:
        # 从文件中读取四维数组
        del_img_data = file['array'][:]
        del_cigar_img = torch.tensor(del_img_data.astype(np.float32))#属于计算密集型

    with h5py.File(n_h5_image_path, 'r') as file:
        # 从文件中读取四维数组
        n_img_data = file['array'][:]
        negative_cigar_img = torch.tensor(n_img_data.astype(np.float32))

    torch.save(ins_cigar_img, ins_pt_image_path)
    torch.save(del_cigar_img, del_pt_image_path)
    torch.save(negative_cigar_img, n_pt_image_path)
    print(f"染色体：{chromosome}的数据类型转换完成。")

def worker_process(chromosome_data):
    # 在子进程中执行 data_type_transfer
    data_type_transfer(chromosome_data)

def main():
    #data_list=[(chromosome, chr_len),(chromosome, chr_len),...]
    data_list=Reference_chromosome_processing()
    all_time_start=time.time()
    for i, chromosome_data in enumerate(data_list):
        one_chr_time_start = time.time()
        One_chromosome_processing(chromosome_data)
        one_chr_time_end = time.time()
        elapsed_one_chr_time = (one_chr_time_end - one_chr_time_start) * 1000
        print(f"处理染色体:{chromosome_data[0]}的总时间：{elapsed_one_chr_time:.3f}ms")
    all_time_end=time.time()
    elapsed_all_time = (all_time_end - all_time_start) * 1000
    print(f"处理染色体的总时间：{elapsed_all_time:.3f}ms")

    # # data_list=[(chromosome, chr_len),(chromosome, chr_len),...]
    # data_list = Reference_chromosome_processing()
    # chromosome_data=data_list[0]
    # all_time_start = time.time()
    # one_chr_time_start = time.time()
    # One_chromosome_processing(chromosome_data)
    # one_chr_time_end = time.time()
    # elapsed_one_chr_time = (one_chr_time_end - one_chr_time_start) * 1000
    # print(f"处理染色体:{chromosome_data[0]}的总时间：{elapsed_one_chr_time:.3f}ms")
    # all_time_end = time.time()
    # elapsed_all_time = (all_time_end - all_time_start) * 1000
    # print(f"处理染色体的总时间：{elapsed_all_time:.3f}ms")

    #数据类型转换
    #1、多线程
    # print(f"正在进行数据类型转换（int->float32->torch.tensor）")
    # data_transfer_time_start = time.time()
    # # 获取CPU的核心数
    # cpu_count = os.cpu_count()
    # thread_num=int(cpu_count/2)
    # thread_num=1
    # with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
    #     futures = []
    #     for i, chromosome_data in enumerate(data_list):
    #         core_index = i % cpu_count
    #         futures.append(
    #             executor.submit(worker_thread, chromosome_data, core_index))
    #     concurrent.futures.wait(futures)
    # data_transfer_time_end = time.time()
    # elapsed_data_transfer_time = (data_transfer_time_end - data_transfer_time_start) * 1000
    # print(f"数据类型转换时间：{elapsed_data_transfer_time:.3f}ms")

    #2、多进程
    # print(f"正在进行数据类型转换（int->float32->torch.tensor）")
    # data_transfer_time_start = time.time()
    # cpu_count = os.cpu_count()
    # num_processes = int(cpu_count/2+1)  # 这里可以根据你的机器核数来设置进程数量
    # # 使用 multiprocessing.Pool 创建一个进程池
    # with multiprocessing.Pool(num_processes) as pool:
    #     # 使用 map 方法将 worker 函数应用到 data_list 的每个元素
    #     pool.map(worker_process, data_list)
    # data_transfer_time_end=time.time()
    # elapsed_data_transfer_time = (data_transfer_time_end - data_transfer_time_start) * 1000
    # print(f"数据类型转换时间：{elapsed_data_transfer_time:.3f}ms")

    #串行
    data_transfer_time_start=time.time()
    for i, chromosome_data in enumerate(data_list):
        data_type_transfer(chromosome_data)
    data_transfer_time_end = time.time()
    elapsed_data_transfer_time = (data_transfer_time_end - data_transfer_time_start) * 1000
    print(f"数据类型转换时间：{elapsed_data_transfer_time:.3f}ms")

if __name__=='__main__':
    main()
