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
import json
import threading
import h5py

torch.multiprocessing.set_sharing_strategy('file_system')

seed_everything(2022)

#金标数据集
bam_data_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
#CLR
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-CLR/"
#ONT
# bam_data_dir="/mnt/HHD_16T_1/Alignment_data/HG002/ONT/"

vcf_data_dir = "../data/"
data_dir = "../data/"

#金标数据集
bam_path = bam_data_dir + "HG002-PacBio-HiFi-minimap2.sorted.bam"
#CLR
# bam_path = bam_data_dir + "HG002-PacBio_CLR-minimap2.sorted.bam"
#ONT
# bam_path = bam_data_dir + "HG002-ONT-minimap2.sorted.bam"
#NA12878
# bam_path = data_dir + "NA12878_S1.bam"
# bam_path = data_dir + "NA12878_NGMLR_sorted.bam"

ins_vcf_filename = vcf_data_dir + "insert_result_data.csv.vcf"
del_vcf_filename = vcf_data_dir + "delete_result_data.csv.vcf"

hight = 224
resize = torchvision.transforms.Resize([hight, hight])


def position(sum_data):
    chromosome, chr_len = sum_data
    #print(f"chromosome:{chromosome},chr_len:{chr_len}")
    ins_position = []
    del_position = []
    n_position = []
    # insert
    insert_result_data = pd.read_csv(ins_vcf_filename, sep="\t", index_col=0)
    insert_chromosome = insert_result_data[insert_result_data["CHROM"] == chromosome]
    row_pos = []
    ins_row_image_name = []  # image_name
    for index, row in insert_chromosome.iterrows():
        row_pos.append(row["POS"])
        ins_row_image_name.append(row["ID"])

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
    del_row_image_name = []
    for index, row in delete_chromosome.iterrows():
        row_pos.append(row["POS"])
        row_end.append(row["END"])
        del_row_image_name.append(row["ID"])

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

    # .txt文件，存储文件名
    txt_image_name_save_path = data_dir + 'image_name/' + chromosome
    ut.mymkdir(txt_image_name_save_path)
    with open(txt_image_name_save_path + '/insert' + '.txt', "w") as txt_file:
        for item in ins_row_image_name:
            txt_file.write(item + "\n")
    with open(txt_image_name_save_path + '/delete' + '.txt', "w") as txt_file:
        for item in del_row_image_name:
            txt_file.write(item + "\n")

    save_path = data_dir + 'position/' + chromosome
    ut.mymkdir(save_path)
    torch.save(ins_position, save_path + '/insert' + '.pt')
    torch.save(del_position, save_path + '/delete' + '.pt')
    torch.save(n_position, save_path + '/negative' + '.pt')


def create_image(sum_data):
    non_empty_sum=0
    chromosome, chr_len = sum_data

    print("deal chromosome " + chromosome)

    #load data时间计算
    load_data_start=time.time()

    ins_position = torch.load(
        data_dir + 'position/' + chromosome + '/insert' + '.pt')
    del_position = torch.load(
        data_dir + 'position/' + chromosome + '/delete' + '.pt')
    n_position = torch.load(data_dir + 'position/' +
                            chromosome + '/negative' + '.pt')
    load_data_end=time.time()

    #print(chromosome + " cigar start")
    save_path = data_dir + 'image/' + chromosome

    if os.path.exists(save_path + '/negative_cigar_new_img' + '.pt'):
        return

    print(f"ins_rows:{len(ins_position)} del_rows:{len(del_position)} n_rows:{len(n_position)}")

    ins_cigar_img = torch.empty(len(ins_position), 1, hight, hight)
    del_cigar_img = torch.empty(len(del_position), 1, hight, hight)
    negative_cigar_img = torch.zeros(len(n_position), 1, hight, hight)#empty改成zeros

    #计算时间计算
    compute_start=time.time()
    chr_open_bam_time=0
    chr_read_time=0
    chr_image_coding_time=0
    chr_total_kernel_cigar_time=0
    for i, b_e in enumerate(ins_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                ins_cigar_img[i],non_empty,open_bam_time,read_time,image_coding_time,total_kernel_cigar_time = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
                non_empty_sum+=non_empty
                chr_open_bam_time+=open_bam_time
                chr_read_time+=read_time
                chr_image_coding_time+=image_coding_time
                chr_total_kernel_cigar_time+=total_kernel_cigar_time
                #print(f"chromosome:{chromosome},begin:{b_e[0]},end:{b_e[1]}")
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal(ins_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])
                           
        #print("===== finish_cigar_img(ins_position) " + chromosome + " index = " + str(i) + "/" + str(len(ins_position)))

    for i, b_e in enumerate(del_position):
        zoom = 1
        fail = 1
        while fail:
            try:
                fail = 0
                del_cigar_img[i],non_empty,open_bam_time,read_time,image_coding_time,total_kernel_cigar_time = ut.cigar_new_img_single_optimal(
                    bam_path, chromosome, b_e[0], b_e[1], zoom)
                non_empty_sum+=non_empty
                chr_open_bam_time += open_bam_time
                chr_read_time += read_time
                chr_image_coding_time += image_coding_time
                chr_total_kernel_cigar_time += total_kernel_cigar_time
            except Exception as e:
                fail = 1
                zoom += 1
                print(e)
                print("Exception cigar_img_single_optimal(del_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])
            
        #print("===== finish_cigar_img(del_position) " + chromosome + " index = " + str(i) + "/" + str(len(del_position)))

    # for i, b_e in enumerate(n_position):
    #     zoom = 1
    #
    #     fail = 1
    #     while fail:
    #         try:
    #             fail = 0
    #             negative_cigar_img[i],non_empty,open_bam_time,read_time,image_coding_time,total_kernel_cigar_time = ut.cigar_new_img_single_optimal(
    #                 bam_path, chromosome, b_e[0], b_e[1], zoom)
    #             non_empty_sum+=non_empty
    #             chr_open_bam_time += open_bam_time
    #             chr_read_time += read_time
    #             chr_image_coding_time += image_coding_time
    #             chr_total_kernel_cigar_time += total_kernel_cigar_time
    #         except Exception as e:
    #             fail = 1
    #             zoom += 1
    #             print(e)
    #             print("Exception cigar_img_single_optimal(neg_position) " + chromosome + " " + str(zoom) + ". The length = ", b_e[1] - b_e[0])


        #print("===== finish_cigar_img(neg_position) " + chromosome + " index = " + str(i) + "/" + str(len(n_position)))
    compute_end=time.time()

    # .h5文件
    h5_python_save_path = data_dir + 'h5_image_python/' + chromosome
    ut.mymkdir(h5_python_save_path)  # 创建文件夹
    # 创建并打开h5文件
    with h5py.File(h5_python_save_path + '/ins_cigar_new_img' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(ins_cigar_img))
    with h5py.File(h5_python_save_path + '/del_cigar_new_img' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(del_cigar_img))
    with h5py.File(h5_python_save_path + '/negative_cigar_new_img' + '.h5', 'w') as file:
        # 将二维数组保存为数据集
        file.create_dataset('array', data=np.array(negative_cigar_img))

    #存储时间计算
    store_start=time.time()
    ut.mymkdir(save_path)
    torch.save(ins_cigar_img, save_path + '/ins_cigar_new_img' + '.pt')
    torch.save(del_cigar_img, save_path + '/del_cigar_new_img' + '.pt')
    torch.save(negative_cigar_img, save_path + '/negative_cigar_new_img' + '.pt')
    store_end=time.time()

    print(f"{chromosome},non_empty:{non_empty_sum}")

    load_data_time=(load_data_end-load_data_start)*1000
    compute_time=(compute_end-compute_start)*1000
    store_data_time=(store_end-store_start)*1000

    thread_id = threading.get_ident()
    data_length=len(ins_position)+len(del_position)+len(n_position)
    print(f"{thread_id},create_image time start\nchromosome:{chromosome}")
    print(f"{thread_id},加载数据时间：{load_data_time:.3f}ms")
    print(f"{thread_id},计算时间：{compute_time:.3f}ms")
    print(f"    {thread_id},打开bam文件的平均时间：{(1.0*chr_open_bam_time/data_length)/1000:.3f}ms")
    print(f"    {thread_id},计算read两个端点信息的平均时间：{(1.0*chr_read_time/data_length)/1000:.3f}ms")
    print(f"    {thread_id},图像编码平均时间：{(1.0*chr_image_coding_time/data_length)/1000:.3f}ms")
    print(f"    {thread_id},kernel_cigar函数消耗的平均时间：{(1.0*chr_total_kernel_cigar_time/data_length)/1000:.3f}ms")
    print(f"{thread_id},存储数据时间：{store_data_time:.3f}ms")
    print(f"{thread_id},create_image time end")

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

    position_time_start=time.time()
    position([args.chr, int(args.len)])
    position_time_end=time.time()

    create_image_time_start=time.time()
    create_image([args.chr, int(args.len)])
    create_image_time_end=time.time()

    position_time=(position_time_end-position_time_start)*1000
    create_image_time=(create_image_time_end-create_image_time_start)*1000

    thread_id = threading.get_ident()
    print(f"{thread_id},position time:{position_time}\n{thread_id},create_image time{create_image_time}")
