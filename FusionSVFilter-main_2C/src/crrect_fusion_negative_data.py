import torch

def fusion_negative_data(chromosome):
    data_dir = "../data/"
    pt_image_path1 = data_dir + 'image_gold/' + chromosome
    pt_image_path2 =data_dir + 'image_FP/' + chromosome

    pt_img1_path=pt_image_path1+"/negative_cigar_new_img.pt"
    pt_img21_path=pt_image_path2+"/ins_cigar_new_img.pt"
    pt_img22_path=pt_image_path2+"/del_cigar_new_img.pt"

    print(f"======deal chromosome {chromosome}======")

    # 1. 加载3个 .pt 文件中的数组
    array_1 = torch.load(pt_img1_path)  # 读取1.pt中的数组
    array_21 = torch.load(pt_img21_path)  # 读取2.pt中的数组
    array_22 = torch.load(pt_img22_path)  # 读取2.pt中的数组

    # 2. 确保这两个数组的维度是 [image_num, 1, 224, 224]
    print(f"Shape of array_1: {array_1.shape}")
    print(f"Shape of array_21: {array_21.shape}")
    print(f"Shape of array_22: {array_22.shape}")

    # 3. 在第一个维度（即image_num）上拼接两个数组
    combined_array = torch.cat((array_1, array_21), dim=0)
    combined_array = torch.cat((combined_array, array_22), dim=0)

    # 4. 打印拼接后的数组形状，确认操作是否成功
    print(f"Shape of combined_array: {combined_array.shape}")

    # 5. 将拼接后的数组保存回 1.pt
    torch.save(combined_array, pt_img1_path)

    print(f"combined_array has been saved in {pt_img1_path}")

def main():
    chr_name_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19', '20', '21', '22', 'X', 'Y']
    for chr_name in chr_name_list:
        fusion_negative_data(chr_name)

if __name__=='__main__':
    main()
