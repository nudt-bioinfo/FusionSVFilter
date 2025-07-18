'''
改变fn.vcf中的ID
'''

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

# 改变VCF文件中的ID
def change_vcf_files_ID(vcf1, vcf_output, tool_name):
    head1=[]
    sv_data1=[]
    # 读取VCF文件头部信息
    with open(vcf1, 'r') as f1:
        for line in f1:
            if line.startswith("#"):
                head1.append(line)
            else:
                sv_data1.append(line)
    print(len(head1))
    print(len(sv_data1))
    new_vcf_data=[]
    # 创建VCF输出文件，先写入头部信息
    with open(vcf_output, 'w') as output_file:
        # 写入head1
        for line in head1:
            output_file.write(line)
        INS_num=0
        DEL_num=0
        OTHER_num=0
        i=1
        sv_data1_length=len(sv_data1)
        for sv1 in sv_data1:
            print(f"{i}/{sv_data1_length}", end='\r')
            i += 1
            record1=parse_vcf_record(sv1)
            if record1['SVTYPE']=='INS':
                new_id=tool_name+'.'+"INS"+'.'+str(INS_num)
                INS_num+=1
                sv1=sv1.replace(record1['ID'],new_id)
                new_vcf_data.append(sv1)
            elif record1['SVTYPE']=='DEL':
                new_id = tool_name + '.' + "DEL" + '.' + str(DEL_num)
                DEL_num += 1
                sv1 = sv1.replace(record1['ID'], new_id)
                new_vcf_data.append(sv1)
            else:
                new_id = tool_name + '.' + "OTHER" + '.' + str(OTHER_num)
                OTHER_num += 1
                sv1 = sv1.replace(record1['ID'], new_id)
                new_vcf_data.append(sv1)
        # 改变结束
        print("change over")
        for record in new_vcf_data:
            output_file.write(record)

    print(f"INS_num:{INS_num},DEL_num:{DEL_num},OTHER_num:{OTHER_num}")

if __name__=='__main__':
    tool_name="SVision"
    data_dir1 = "/home/zxh/D/program/vcf_data/"
    vcf1_name = "SVision/output_file/fn.vcf"
    output_file = "output_FN_analysis/SVision_FN.vcf"
    vcf1_path = data_dir1+vcf1_name
    vcf_output_path = data_dir1+output_file
    change_vcf_files_ID(vcf1_path, vcf_output_path, tool_name)