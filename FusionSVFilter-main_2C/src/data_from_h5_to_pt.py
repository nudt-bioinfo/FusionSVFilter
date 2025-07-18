import argparse
import subprocess
import concurrent.futures
from pytorch_lightning import seed_everything
import torch
import h5py
import numpy as np

def parse_args():
    """
    :return:进行参数的解析
    """
    description = "you should add those parameter"
    parser = argparse.ArgumentParser(description=description)
    help = "The path of address"
    parser.add_argument('--chr', help=help)
    parser.add_argument('--len', help=help)
    #接下来就可以通过args.chr和args.len提取两个变量的值
    args = parser.parse_args()
    return args

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


if __name__ == '__main__':
    args = parse_args()
    data_type_transfer([args.chr, int(args.len)])

