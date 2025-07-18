import pandas as pd
import re

data_dir = "../data/"

def list_save(filename, data):
    file = open(filename,'w')
    file.writelines(data)
    file.close()
    print(filename + " file saved successfully")

def set_save(filename, data):
    file = open(filename,'w')
    file.writelines([line+'\n' for line in data])
    file.close()
    print(filename + " file saved successfully")

insert = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG002\n"]
delete = ["CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG002\n"]

# filename = data_dir + "HG002_SVs_Tier1_v0.6_PASS.vcf"
# filename = data_dir + "cutesv-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "cuteSV_PBSV-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "PBSV_different_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "cuteSV_SVision-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "PBSV_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "Sniffles_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "SVIM_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "SVision_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "cuteSV_PBSV_Sniffles_SVIM_SVision_fusion.sorted.vcf"
# filename = data_dir + "pbsv-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "sniffles-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "svim-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "svision-HiFi-pbmm2.sorted.vcf"
# filename = data_dir + "cutesv-HiFi-minimap2.sorted.vcf"
# filename = data_dir + "sniffles-HiFi-minimap2.sorted.vcf"
# filename = data_dir + "cutesv-CLR-pbmm2.sorted.vcf"
# filename = data_dir + "sniffles-CLR-pbmm2.sorted.vcf"
filename = data_dir + "svim-CLR-pbmm2.sorted.vcf"
# filename = data_dir + "pbsv-CLR-pbmm2.sorted.vcf"
# filename = data_dir + "svision-CLR-pbmm2.sorted.vcf"
# filename = data_dir + "cuteSV_PBSV_Sniffles_SVIM_SVision_FP_fusion.sorted.vcf"
# filename = data_dir + "cutesv-ONT-minimap2.sorted.vcf"
# filename = data_dir + "pbsv-ONT-pbmm2.sorted.vcf"
# filename = data_dir + "sniffles-ONT-minimap2.sorted.vcf"
# filename = data_dir + "svim-ONT-minimap2.sorted.vcf"
# filename = data_dir + "svision-ONT-minimap2.sorted.vcf"
# filename = data_dir + "cuteSV_PBSV_Sniffles_SVIM_SVision_fusion_CLR_pbmm2_imprecise.sorted.vcf"
## CLR fusion
# filename = data_dir + "PBSV_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "Sniffles_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "SVIM_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "SVision_difference_cmp_to_cuteSV.sorted.vcf"
# filename = data_dir + "cuteSV_Sniffles_SVIM_SVision_fusion_CLR_pbmm2_precise.sorted.vcf"
# filename = data_dir + "cuteSV_Sniffles_SVIM_SVision_PBSV_fusion_CLR_pbmm2_precise.sorted.vcf"
# filename = data_dir + "cuteSV_SVIM_SVision_PBSV_fusion_CLR_pbmm2_precise.sorted.vcf"
## ONT fusion
# filename = data_dir + "cuteSV_Sniffles_SVIM_SVision_fusion_ONT_minimap2_precise.sorted.vcf"
# filename = data_dir + "cuteSV_Sniffles_SVIM_SVision_PBSV_fusion_ONT_minimap2_precise.sorted.vcf"
# filename = data_dir + "cuteSV_Sniffles_PBSV_SVIM_fusion_ONT_minimap2_precise.sorted.vcf"
# filename = data_dir + "cuteSV_Sniffles_SVIM_fusion_ONT_minimap2_precise.sorted.vcf"

chr_list = set()

with open(filename, "r") as f:
    lines = f.readlines()
    for data in lines:
        if "#" in data:
            if "contig=<ID=" in data:
                chr_list.add(re.split("=|,", data)[2])
        else:
            if "SVTYPE=DEL" in data:
                delete.append(data)
            elif "SVTYPE=INS" in data:
                insert.append(data)

list_save(filename + "_ins", insert)
list_save(filename + "_del", delete)
set_save(filename + "_chr", chr_list)
print(f"len(ins):{len(insert)},len(del):{len(delete)}")

insert_result_data = pd.read_csv(filename + "_ins", sep = "\t")
print(f"测试insert_result_data的大小1：{len(insert_result_data)}")

insert_result_data.insert(2,'SPOS',0)
insert_result_data.insert(3,'EPOS',0)
insert_result_data.insert(4,'SVLEN',0)
print(f"测试insert_result_data的大小2：{len(insert_result_data)}")

for index, row in insert_result_data.iterrows():
    print(f"INS index = {index}", end='\r')
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        insert_result_data.loc[index, ["SPOS"]] = start
        insert_result_data.loc[index, ["EPOS"]] = end
    else:
        insert_result_data.loc[index, ["SPOS"]] = 0
        insert_result_data.loc[index, ["EPOS"]] = 0

    s = row["INFO"]
    pos = s.find("SVLEN")
    if pos == -1:
        pos = s.find("END") + 4
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s) - row["POS"]
        insert_result_data.loc[index, ["SVLEN"]] = s
    else:
        pos = pos + 6
        s = s[pos:]
        s = s.split(";")[0]
        s = int(s)
        insert_result_data.loc[index, ["SVLEN"]] = s
print(f"测试insert_result_data的大小3：{len(insert_result_data)}")
insert_result_data.to_csv(data_dir + "insert_result_data.csv.vcf", sep="\t")

print()
print(f"INS finished, total number = {index+1}")

delete_result_data = pd.read_csv(filename + "_del", sep = "\t")

delete_result_data.insert(2,'SPOS',0)
delete_result_data.insert(3,'EPOS',0)
delete_result_data.insert(4,'END',0)
delete_result_data.insert(5,'SEND',0)
delete_result_data.insert(6,'EEND',0)

for index, row in delete_result_data.iterrows():
    print(f"DEL index = {index}", end='\r')
    s = row["INFO"]
    pos = s.find("CIPOS")
    if pos != -1:
        pos = pos + 6 
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SPOS"]] = start
        delete_result_data.loc[index, ["EPOS"]] = end
    else:
        delete_result_data.loc[index, ["SPOS"]] = 0
        delete_result_data.loc[index, ["EPOS"]] = 0

    s = row["INFO"]
    pos = s.find("END")
    if pos != -1:
        pos += 4
        s_end = s[pos:]
        s_end = s_end.split(";")[0]
        try:
            end_value = int(s_end)
        except ValueError:
            print(f"Error parsing END value: {s_end}")
            end_value = None
    else:
        pos = s.find("SVLEN")
        if pos != -1:
            pos +=6
            s_svlen = s[pos:]
            s_svlen = s_svlen.split(";")[0]
            try:
                svlen_value = abs(int(s_svlen))
                end_value = row["POS"] + svlen_value
                value_tmp = row["POS"]
            except ValueError:
                print(f"Error parsing SVLEN value: {s_svlen}")
                end_value = None
        else:
            print(f"Neither END nor SVLEN found in INFO: {s}")
            end_value = None

    if end_value is not None:
        delete_result_data.loc[index, ["END"]] = end_value
    else:
        print(f"Unable to set END for index {index}")

    s = row["INFO"]
    pos = s.find("CIEND")
    if pos != -1:
        pos = pos + 6 
        s = s[pos:]
        s = s.split(";")[0]
        s = s.split(",")
        start = int(s[0])
        end = int(s[1])
        delete_result_data.loc[index, ["SEND"]] = start
        delete_result_data.loc[index, ["EEND"]] = end
    else:
        delete_result_data.loc[index, ["SEND"]] = 0
        delete_result_data.loc[index, ["EEND"]] = 0

delete_result_data.to_csv(data_dir + "delete_result_data.csv.vcf", sep="\t")
print()
print(f"len(ins_image):{len(insert_result_data)},len(del_image):{len(delete_result_data)}")
print(f"DEL finished, total number = {index+1}")
