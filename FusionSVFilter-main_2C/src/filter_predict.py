import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interp
from tqdm import tqdm
import numpy as np
from filter_net import IDENet

# 对于过滤，需要自定义数据集类，用于加载需要过滤的数据
class PredictDataset(Dataset):
    def __init__(self, path="../data/", transform=None):
        self.insfile_list = os.listdir(path + "ins")
        self.delfile_list = os.listdir(path + "del")
        self.path = path
        self.transform = transform
        self._len = len(self.insfile_list) + len(self.delfile_list)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if index < len(self.insfile_list):
            x, y = torch.load(self.path + "ins/" + str(index) + ".pt")
            x = self.transform(x)
            return x, y
        elif index < len(self.insfile_list) + len(self.delfile_list):
            index -= len(self.insfile_list)
            x, y = torch.load(self.path + "del/" + str(index) + ".pt")
            x = self.transform(x)
            return x, y
        else:
            print("error:no n img data")#过滤的时候不需要nagative数据集

def prepare_data(data_path="../data/", train_size=0):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.458, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = PredictDataset(data_path, transform=transform)
    dataset_size = len(dataset)
    print(f"the number of filtering data is {dataset_size}")
    return dataset


def get_model(data_dir="../data/", config=None):
    # Instantiate the model with the config parameters
    model = IDENet(path=data_dir, config=config)
    return model


# 预测过程
def predict_model():
    data_dir="../data/"
    config = {
        "lr": 7.1873e-06,
        "batch_size": 64,
        "beta1": 0.9,
        "beta2": 0.999,
        'weight_decay': 0.0011615,
        'model': "ResNet50"
    }

    vcf_data_dir = "../data/"
    # vcf_name = "HG002_SVs_Tier1_v0.6_PASS"
    # vcf_name = "cutesv-HiFi-pbmm2.sorted"
    # vcf_name = "cuteSV_PBSV-HiFi-pbmm2.sorted"
    # vcf_name = "PBSV_different_cmp_to_cuteSV.sorted"
    # vcf_name = "cuteSV_SVision-HiFi-pbmm2.sorted"
    # vcf_name = "PBSV_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "Sniffles_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "SVIM_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "SVision_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "cuteSV_PBSV_Sniffles_SVIM_SVision_fusion.sorted"
    # vcf_name = "pbsv-HiFi-pbmm2.sorted"
    # vcf_name = "sniffles-HiFi-pbmm2.sorted"
    # vcf_name = "svim-HiFi-pbmm2.sorted"
    # vcf_name = "svision-HiFi-pbmm2.sorted"
    # vcf_name = "cutesv-HiFi-minimap2.sorted"
    # vcf_name = "sniffles-HiFi-minimap2.sorted"
    # vcf_name = "cutesv-CLR-pbmm2.sorted"
    # vcf_name = "sniffles-CLR-pbmm2.sorted"
    vcf_name = "svim-CLR-pbmm2.sorted"
    # vcf_name = "pbsv-CLR-pbmm2.sorted"
    # vcf_name = "svision-CLR-pbmm2.sorted"
    # vcf_name = "cutesv-ONT-minimap2.sorted"
    # vcf_name = "pbsv-ONT-pbmm2.sorted"
    # vcf_name = "sniffles-ONT-minimap2.sorted"
    # vcf_name = "svim-ONT-minimap2.sorted"
    # vcf_name = "svision-ONT-minimap2.sorted"
    # vcf_name = "cuteSV_PBSV_Sniffles_SVIM_SVision_fusion_CLR_pbmm2_imprecise.sorted"
    ## CLR fusion
    # vcf_name = "PBSV_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "Sniffles_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "SVIM_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "SVision_difference_cmp_to_cuteSV.sorted"
    # vcf_name = "cuteSV_Sniffles_SVIM_SVision_fusion_CLR_pbmm2_precise.sorted"
    # vcf_name = "cuteSV_Sniffles_SVIM_SVision_PBSV_fusion_CLR_pbmm2_precise.sorted"
    # vcf_name = "cuteSV_SVIM_SVision_PBSV_fusion_CLR_pbmm2_precise.sorted"
    ## ONT fusion
    # vcf_name = "cuteSV_Sniffles_SVIM_SVision_fusion_ONT_minimap2_precise.sorted"
    # vcf_name = "cuteSV_Sniffles_SVIM_SVision_PBSV_fusion_ONT_minimap2_precise.sorted"
    # vcf_name = "cuteSV_Sniffles_PBSV_SVIM_fusion_ONT_minimap2_precise.sorted"
    # vcf_name = "cuteSV_Sniffles_SVIM_fusion_ONT_minimap2_precise.sorted"

    # 加载预训练好的模型
    model = get_model(data_dir,config=config)
    model_dir1="../models-self/"
    model_dir2="../model_save_dir/"
    model_name1="ResNet50_2C_96"
    # ResNet50_2C_FP_crrect_0 ResNet50_2C_FP_0_96.ckpt
    # ResNet50_2C_FP_crrect_10 ResNet50_2C_FP_10_96.ckpt
    # ResNet50_2C_FP_crrect_15 ResNet50_2C_FP_15_97.ckpt
    model_name2="ResNet50_2C_FP_crrect_10/ResNet50_2C_FP_10_96"
    model_path=model_dir1+model_name1
    # 加载.ckpt文件
    checkpoint = torch.load(model_path+'.ckpt', map_location='cpu')  # 使用map_location='cpu'确保可以加载到CPU上
    # 假设.ckpt文件中保存了'state_dict'（如果有其他结构，可能需要适当修改）
    model.load_state_dict(checkpoint['state_dict'])
    print(f"{model_path}模型加载成功")

    model.eval()

    # 设定设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 准备数据
    predict_path = '../data/'  # 假设需要过滤的数据在'../data/'中
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.458, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = PredictDataset(predict_path, transform=transform)
    predict_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []

    ins_df = pd.read_csv(vcf_data_dir + vcf_name + ".vcf_ins", sep="\t", dtype={0:str})
    del_df = pd.read_csv(vcf_data_dir + vcf_name + ".vcf_del", sep="\t", dtype={0:str})
    # ins_df[0].astype(str)
    # ins_df[1].astype(int)
    # ins_df[2].astype(str)
    # del_df[0].astype(str)
    filtered_ins_df = pd.DataFrame(columns=ins_df.columns)
    filtered_del_df = pd.DataFrame(columns=del_df.columns)

    print(f"insert SV number:{len(ins_df)},delete SV number:{len(del_df)}")

    with open(data_dir + '/all_ins_image_name.txt', "r") as txt_file:
        all_ins_image_name = txt_file.read().splitlines()
    with open(data_dir + '/all_del_image_name.txt', "r") as txt_file:
        all_del_image_name = txt_file.read().splitlines()

    predict_loader_size = len(predict_loader)

    assert len(all_ins_image_name) + len(all_del_image_name) == predict_loader_size, \
        "SV图片的数量不等于SV图片名字的数量，这会导致图片和图片名字不能一一对应。"

    print(f"insert_SV_image number:{len(all_ins_image_name)}\ndelete_SV_image number:{len(all_del_image_name)}")
    print(f"为{len(all_ins_image_name)+len(all_del_image_name)}张图片生成标签")

    i = 0
    # 预测每张图片的标签
    with torch.no_grad():
        for inputs, _ in predict_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)#outputs示例：tensor([[0.7163, 0.2837]], device='cuda:0')
            # 将张量从GPU移动到CPU，并取出数值
            output_1 = outputs[0, 0].item()  # 获取第一个数字
            output_2 = outputs[0, 1].item()  # 获取第二个数字
            if output_2>output_1 and output_2<0.5:
                print("output_2>output_1 and output_2<0.5")
            # if i<10:
            #     print(outputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            print(f"predict index {i+1}/{predict_loader_size}", end='\r')
            i += 1
    print()

    predicted_labels = [int(pred[0]) for pred in all_preds]  # 每个预测结果

    # 将预测标签存储到文件
    with open('filter_predicted_labels.txt', 'w') as f:
        for idx, label in enumerate(predicted_labels):
            # f.write(f"Index {idx}: Predicted label: {label}\n")
            f.write(f"{label}\n")
    print(f"{predict_loader_size}张图片的预测标签已存储至filter_predicted_labels.txt中")

    print(f"filter insert SV start")
    insert_SV_number=len(ins_df)
    for index,sv_data in ins_df.iterrows():
        print(f"insert_sv index {index+1}/{insert_SV_number}",end='\r')
        sv_id=sv_data["ID"]
        if sv_id in all_ins_image_name:
            img_index=all_ins_image_name.index(sv_id)
            predicted_label = predicted_labels[img_index]
            if predicted_label == 0:
                # filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
                continue
            elif predicted_label == 1:
                filtered_ins_df = filtered_ins_df.append(ins_df.iloc[index])

        # else:
        #     filtered_ins_df = filtered_ins_df.append(ins_df.iloc[index])
    print()
    print(f"filter insert SV end")

    print(f"filter delete SV start")
    delete_SV_number = len(del_df)
    for index,sv_data in del_df.iterrows():
        print(f"delete_sv index {index + 1}/{delete_SV_number}", end='\r')
        sv_id=sv_data["ID"]
        if sv_id in all_del_image_name:
            img_index=all_del_image_name.index(sv_id)
            img_index=img_index+len(all_ins_image_name)
            predicted_label = predicted_labels[img_index]
            if predicted_label == 0:
                # filtered_del_df = filtered_del_df.append(del_df.iloc[i])
                continue
            elif predicted_label == 1:
                filtered_del_df = filtered_del_df.append(del_df.iloc[index])
        # else:
        #     filtered_del_df = filtered_del_df.append(del_df.iloc[index])
    print()
    print(f"filter delete SV end")
    
    # for index in range(len(ins_df)):
    #     predicted_label = predicted_labels[index]
    #     if predicted_label == 0:
    #         # filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])
    #         continue
    #     elif predicted_label == 1:
    #         # temp_row = ins_df.iloc[i].copy()
    #         # temp_row['ALT'] = '<DEL>'
    #         # temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=INS', 'SVTYPE=DEL')
    #         # filtered_ins_df = filtered_ins_df.append(temp_row)
    #         continue
    #     elif predicted_label == 2:
    #         filtered_ins_df = filtered_ins_df.append(ins_df.iloc[i])

    # for index in range(len(del_df)):
    #     index=index+len(ins_df)
    #     predicted_label = predicted_labels[index]
    #     if predicted_label == 0:
    #         # filtered_del_df = filtered_del_df.append(del_df.iloc[i])
    #         continue
    #     elif predicted_label == 1:
    #         filtered_del_df = filtered_del_df.append(del_df.iloc[i])
    #     elif predicted_label == 2:
    #         # temp_row = del_df.iloc[i].copy()
    #         # temp_row['ALT'] = '<INS>'
    #         # temp_row['INFO'] = temp_row['INFO'].replace('SVTYPE=DEL', 'SVTYPE=INS')
    #         # filtered_del_df = filtered_del_df.append(temp_row)
    #         continue

    filtered_ins_df.to_csv(vcf_data_dir + vcf_name + "_filtered.vcf_ins", sep="\t", index=False)
    filtered_del_df.to_csv(vcf_data_dir + vcf_name + "_filtered.vcf_del", sep="\t", index=False)

    with open(vcf_data_dir + vcf_name + ".vcf") as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith("#")]
    data_lines = [line for line in lines if not line.startswith("#")]

    with open(data_dir + "data_temp.vcf", "w") as f:
        for line in data_lines:
            f.write(line)

    vcf_df = pd.read_csv(data_dir + "data_temp.vcf", sep="\t", header=None, dtype={0:str})#数据以制表符分隔，header=None表示文件没有表头
    ins_file_path = vcf_data_dir + vcf_name + "_filtered.vcf_ins"
    try:
        filtered_ins_df = pd.read_csv(ins_file_path, sep="\t", skiprows=1, header=None, dtype={0:str})#skiprows=1表示跳过第一行
    except pd.errors.EmptyDataError:
        filtered_ins_df = pd.DataFrame()
    filtered_del_df = pd.read_csv(vcf_data_dir + vcf_name + "_filtered.vcf_del", sep="\t", skiprows=1, header=None, dtype={0:str})
    # vcf_df[0].astype(str)
    # filtered_ins_df[0].astype(str)
    # filtered_del_df[0].astype(str)
    filtered_df = pd.concat([filtered_ins_df, filtered_del_df])

    vcf_df['index'] = vcf_df[0].astype(str) + "_" + vcf_df[1].astype(str) + "_" + vcf_df[2].astype(str)
    filtered_df['index'] = filtered_df[0].astype(str) + "_" + filtered_df[1].astype(str) + "_" + filtered_df[2].astype(str)

    indices_to_keep = set(filtered_df['index'].values)
    vcf_df_filtered = pd.DataFrame()

    print("indices_to_keep:")
    print(list(indices_to_keep)[0])

    print("vcf_df.iloc[i]['index']:")
    print(vcf_df.iloc[0]['index'])

    print(f"len(filtered_ins_df)+len(filtered_del_df)={len(filtered_ins_df)+len(filtered_del_df)}")
    print(f"len(indices_to_keep):{len(indices_to_keep)}")

    print("正在生成过滤结果")
    selected_num=0
    for i in tqdm(range(len(vcf_df))):
        if vcf_df.iloc[i]['index'] in indices_to_keep:
            selected_num+=1
            vcf_df_filtered = vcf_df_filtered.append(vcf_df.iloc[i])
    print(f"selected_num:{selected_num}")

    import os
    os.remove(data_dir + "data_temp.vcf")

    vcf_df_filtered = vcf_df_filtered.drop(columns=['index'])

    vcf_df_filtered[1] = vcf_df_filtered[1].astype(int)

    with open(data_dir + vcf_name + "_INS_DEL_filtered.vcf", "w") as f:
        for line in header_lines:
            f.write(line)
        vcf_df_filtered.to_csv(f, sep="\t", header=False, index=False)

if __name__ == "__main__":
    predict_model()
