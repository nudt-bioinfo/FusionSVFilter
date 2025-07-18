'''
不处理ID，速度比较慢，但是比较精确
'''
# -*- coding: utf-8 -*-
import os

# 定义SV比较的容忍范围，单位：碱基对
POSITION_TOLERANCE = 500  # 容忍的最大位置差异
LENGTH_TOLERANCE = 100  # 容忍的最大长度差异


# 解析VCF记录
def parse_vcf_record(line):
    data_list = line.strip().split('\t')
    chrom = data_list[0]
    pos = int(data_list[1])
    id=data_list[2]
    info = data_list[7]

    # 获取SVTYPE和SVLEN
    sv_type = None
    sv_len = None

    for field in info.split(';'):
        if field.startswith('SVTYPE='):
            sv_type = field.split('=')[1]
        elif field.startswith('SVLEN='):
            sv_len = int(field.split('=')[1])
            sv_len = abs(sv_len)

    return {
        'CHROM': chrom,
        'POS': pos,
        'ID':id,
        'SVTYPE': sv_type,
        'SVLEN': sv_len
    }

# 判断两个SV是否相同
def is_same_sv(record1, record2, only_ins_del=False):
    if record1['CHROM'] != record2['CHROM']:
        return False

    if record1['SVTYPE'] != record2['SVTYPE']:
        return False

    if only_ins_del and record1['SVTYPE'] not in ['INS', 'DEL']:
        return False

    if abs(record1['POS'] - record2['POS']) > POSITION_TOLERANCE:
        return False

    svlen1 = record1.get('SVLEN', None)
    svlen2 = record2.get('SVLEN', None)

    if svlen1 is not None and svlen2 is not None:
        if abs(svlen1 - svlen2) > LENGTH_TOLERANCE:
            return False

    return True

# 合并VCF文件
def merge_vcf_files(vcf1, vcf2, vcf_output):
    head1=[]
    head2=[]
    sv_data1=[]
    sv_data2=[]
    # 读取VCF文件头部信息
    with open(vcf1, 'r') as f1:
        for line in f1:
            if line.startswith("#"):
                head1.append(line)
            else:
                sv_data1.append(line)
    head_index=head1[-1]
    head1.pop()
    with open(vcf2, 'r') as f2:
        for line in f2:
            if line.startswith("#"):
                head2.append(line)
            else:
                sv_data2.append(line)
    head2.pop()
    print(f"VCF 1头部长度：{len(head1)}")
    print(f"VCF 1结构变异数量：{len(sv_data1)}")
    print(f"VCF 2头部长度：{len(head2)}")
    print(f"VCF 2结构变异数量：{len(sv_data2)}")
    # 创建VCF输出文件，先写入头部信息
    with open(vcf_output, 'w') as output_file:
        # 写入head1
        for line in head1:
            output_file.write(line)

        # 将head2中没有出现在head1中的内容写入vcf_output
        seen_header = set(line.strip() for line in head1)
        for line in head2:
            if line.strip() not in seen_header:
                output_file.write(line)
        output_file.write(head_index)

        for sv1 in sv_data1:
            output_file.write(sv1)

        # 合并SV数据
        sv_data1_length = len(sv_data1)
        sv_data2_length = len(sv_data2)
        i = 1
        sv2_i = 0
        same_pre=-1
        same_cur=-1
        difference1=0
        difference2=0
        difference2_list=[]
        same=0
        for sv1 in sv_data1:
            print(f"{i}/{sv_data1_length}", end='\r')
            i+=1
            record1 = parse_vcf_record(sv1)
            is_merged = False
            sv2_i=same_cur+1
            while(True):
                if sv2_i>=sv_data2_length:
                    break
                sv2=sv_data2[sv2_i]
                record2 = parse_vcf_record(sv2)
                if is_same_sv(record1, record2):
                    same+=1
                    is_merged = True
                    same_pre=same_cur
                    same_cur=sv2_i
                    for ii in range(same_pre+1,same_cur):
                        difference2_list.append(sv_data2[ii])
                        difference2+=1
                    break
                else:#不相同
                    sv2_i+=1
            if is_merged==False:
                difference1+=1
        same_cur+=1
        while(same_cur<sv_data2_length):
            difference2_list.append(sv_data2[same_cur])
            same_cur+=1
            difference2+=1

        # 将合并后的记录写入文件
        for record in difference2_list:
            output_file.write(record)

    print(f"{same},{difference1},{difference2}")

if __name__=='__main__':
    # cuteSV PBSV Sniffles SVIM SVision
    data_dir1 = "/home/zxh/D/program/vcf_data/"
    data_dir2 = "../data/"
    vcf1_name = "vcf_5_tools/ONT_minimap2/merge_vcf_files_4_difference_ResNet50_2C_96_2_len/cuteSV_Sniffles_fusion.sorted.vcf"
    vcf2_name = "vcf_5_tools/ONT_minimap2/SVIM_difference_cmp_to_cuteSV_INS_DEL_filtered.sorted.vcf"
    output_file = "vcf_5_tools/ONT_minimap2/merge_vcf_files_4_difference_ResNet50_2C_96_2_len/cuteSV_Sniffles_SVIM_fusion.vcf"
    vcf1_path = data_dir2+vcf1_name
    vcf2_path = data_dir2+vcf2_name
    vcf_output_path = data_dir2+output_file
    print("正在处理：")
    print(f"{vcf1_path}")
    print(f"{vcf2_path}")
    print(f"{vcf_output_path}")
    merge_vcf_files(vcf1_path, vcf2_path, vcf_output_path)