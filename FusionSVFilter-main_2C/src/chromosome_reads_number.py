import json
import numpy as np


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


def calculate_mean_and_variance(data):
    if not data:
        return None, None  # 如果列表为空，返回 None, None

    # 计算平均值
    mean = sum(data) / len(data)

    # 计算方差
    variance = sum((x - mean) ** 2 for x in data) / len(data)

    return mean, variance


def main():
    chromosome_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
                       "18", "19", "20", "21", "22", "X", "Y"]
    data_dir = "../check_bam_file_data/" + "ONT_"
    all_read_num=0
    chr_all_read_num_list=[]
    for chromosome in chromosome_list:
        ins_all_dict, del_all_dict, n_all_dict = load_from_json(data_dir)
        ins_dict = ins_all_dict[chromosome]
        del_dict = del_all_dict[chromosome]
        n_dict = n_all_dict[chromosome]

        ins_read_num_list = ins_dict["read_lines"]
        del_read_num_list = del_dict["read_lines"]
        n_read_num_list = n_dict["read_lines"]

        chr_all_read_num=0

        ins_read_num=np.sum(ins_read_num_list)
        del_read_num = np.sum(del_read_num_list)
        n_read_num = np.sum(n_read_num_list)

        chr_all_read_num+=ins_read_num+del_read_num+n_read_num
        all_read_num+=ins_read_num+del_read_num+n_read_num
        chr_all_read_num_list.append(chr_all_read_num)

        ins_mean, ins_variance=calculate_mean_and_variance(ins_read_num_list)
        del_mean, del_variance = calculate_mean_and_variance(del_read_num_list)
        n_mean, n_variance = calculate_mean_and_variance(n_read_num_list)

        print(f"{chr_all_read_num}")
        # print(f"{chromosome}:chr_all_read_num:{chr_all_read_num}")
        # print(f"{chromosome}:{ins_mean:.0f},{ins_variance:.0f};{del_mean:.0f},{del_variance:.0f};{n_mean:.0f},{n_variance:.0f}")
        # print(f"{chromosome}:chr_all_read_num:{chr_all_read_num},ins_read_num:{np.sum(ins_read_num_list)},del_read_num:{np.sum(del_read_num_list)},n_read_num:{np.sum(n_read_num_list)}")
    arr=np.array(chr_all_read_num_list)
    max_read_num=np.max(arr)
    min_read_num=np.min(arr)
    mean_read_num=np.mean(arr)
    print(f"all_read_num:{all_read_num},max_read_num:{max_read_num},min_read_num:{min_read_num},mean_read_num:{mean_read_num}")

if __name__ == '__main__':
    main()