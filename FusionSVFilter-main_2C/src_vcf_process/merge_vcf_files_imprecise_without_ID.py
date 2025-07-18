'''
不处理ID，快速合并，但是会有一定的误差
'''
# -*- coding: utf-8 -*-
import os

# 定义SV比较的容忍范围，单位：碱基对
POSITION_TOLERANCE = 500  # 容忍的最大位置差异
LENGTH_TOLERANCE = 100  # 容忍的最大长度差异
chromosome_list=[]


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
    if chrom not in chromosome_list:
        chromosome_list.append(chrom)
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

        #################################
        chromosome_list1 = []
        for sv1 in sv_data1:
            record1=parse_vcf_record(sv1)
            chrom=record1['CHROM']
            if chrom not in chromosome_list1:
                chromosome_list1.append(chrom)
        print(f"VCF 1中染色体出现的顺序：{chromosome_list1}")
        chromosome_list2=[]
        for sv2 in sv_data2:
            record2=parse_vcf_record(sv2)
            chrom=record2['CHROM']
            if chrom not in chromosome_list2:
                chromosome_list2.append(chrom)
        print(f"VCF 2中染色体出现的顺序：{chromosome_list2}")
        print(f"若染色体出现的顺序相差较大，请使用精确合并VCF文件的python代码")
        same=0
        dif1=0
        dif2=0
        merge_sv_list=[]
        sv_data1_length = len(sv_data1)
        sv_data2_length = len(sv_data2)
        sv1_i=0
        sv2_j=0
        while(sv1_i<sv_data1_length and sv2_j<sv_data2_length):
            print(f"{sv1_i}/{sv_data1_length} {sv2_j}/{sv_data2_length}", end='\r')
            sv1=sv_data1[sv1_i]
            sv2=sv_data2[sv2_j]
            record1 = parse_vcf_record(sv1)
            record2 = parse_vcf_record(sv2)

            if is_same_sv(record1, record2):
                merge_sv_list.append(sv1)
                sv1_i+=1
                sv2_j+=1
                same+=1
            else:
                if record1['CHROM']==record2['CHROM']:
                    if record1['POS']<record2['POS']:
                        merge_sv_list.append(sv1)
                        sv1_i+=1
                        dif1+=1
                    else:
                        merge_sv_list.append(sv2)
                        sv2_j+=1
                        dif2+=1
                else:
                    sv1_chr_index=chromosome_list.index(record1['CHROM'])
                    sv2_chr_index=chromosome_list.index(record2['CHROM'])
                    if sv1_chr_index<sv2_chr_index:
                        merge_sv_list.append(sv1)
                        sv1_i += 1
                        dif1 += 1
                    else:
                        merge_sv_list.append(sv2)
                        sv2_j += 1
                        dif2 += 1

        while(sv1_i<sv_data1_length):
            print(f"{sv1_i}/{sv_data1_length} {sv2_j}/{sv_data2_length}", end='\r')
            merge_sv_list.append(sv_data1[sv1_i])
            sv1_i+=1
            dif1 += 1
        while(sv2_j<sv_data2_length):
            print(f"{sv1_i}/{sv_data1_length} {sv2_j}/{sv_data2_length}", end='\r')
            merge_sv_list.append(sv_data2[sv2_j])
            sv2_j+=1
            dif2 += 1
        # 将合并后的记录写入文件
        for sv in merge_sv_list:
            output_file.write(sv)
        print(f"\n{same},{dif1},{dif2}")
        print(len(merge_sv_list))
        print(f"实际运行过程中染色体出现的顺序：{chromosome_list}")
        ################################

if __name__=='__main__':
    # cuteSV PBSV Sniffles SVIM SVision
    data_dir1 = "/home/zxh/D/program/vcf_data/"
    data_dir2 = "../data/vcf_5_tools/ONT_minimap2/"
    vcf1_name = "output/cuteSV_PBSV_Sniffles_SVIM_fusion_ONT_minimap2_imprecise.sorted.vcf"
    vcf2_name = "svision-ONT-minimap2.sorted.vcf"
    output_file = "output/cuteSV_PBSV_Sniffles_SVIM_SVision_fusion_ONT_minimap2_imprecise.vcf"
    vcf1_path = data_dir2+vcf1_name
    vcf2_path = data_dir2+vcf2_name
    vcf_output_path = data_dir2+output_file
    merge_vcf_files(vcf1_path, vcf2_path, vcf_output_path)