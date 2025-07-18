import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torchvision import transforms
from PIL import Image
import shutil

# Helper functions
def load_images_from_pt(file_path):
    """ Load .pt file, which contains a tensor of shape [image_num, 1, 224, 224] """
    data = torch.load(file_path)
    return data.squeeze(1)  # Removing the channel dimension to get [image_num, 224, 224]

def pca_transform(images, n_components=50):
    """ Perform PCA on the images and return the top n_components principal components """
    # Flatten each image to a 1D vector
    images_flat = images.view(images.size(0), -1)  # Flatten to [num_images, 224*224]
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images_flat.numpy())
    return pca_result

# 两个矩阵求相似度，输出一维向量
def compute_similarity1(pca1, pca2):
    """ Compute the cosine similarity between two PCA results """
    return np.dot(pca1, pca2.T) / (np.linalg.norm(pca1) * np.linalg.norm(pca2))

#两个向量求相似度
def compute_cosine_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)  # 点积
    norm1 = np.linalg.norm(vec1)  # 向量1的范数
    norm2 = np.linalg.norm(vec2)  # 向量2的范数
    similarity = dot_product / (norm1 * norm2)  # 余弦相似度
    return similarity

#两个矩阵求相似度，返回一个常量值
def compute_similarity2(mat1, mat2):
    # 计算矩阵的总体余弦相似度，返回一个常量值
    row_similarities = []

    # 对每一行计算余弦相似度
    for row1, row2 in zip(mat1, mat2):
        similarity = compute_cosine_similarity(row1, row2)
        row_similarities.append(similarity)
    # 将所有行的相似度进行平均，得到一个常量值
    return np.mean(row_similarities)

#两个向量求相似度
def compute_similarity(vec1, vec2):
    # 计算两个向量的余弦相似度
    dot_product = np.dot(vec1, vec2)  # 点积
    norm1 = np.linalg.norm(vec1)  # 向量1的范数
    norm2 = np.linalg.norm(vec2)  # 向量2的范数
    similarity = dot_product / (norm1 * norm2)  # 余弦相似度
    return similarity

# Main function
def process_images(chromosome):
    print(f"==========deal chromosome {chromosome}===========")
    # File paths
    data_dir="../data/"
    pt_image_path = data_dir+'image/'+chromosome
    negative_path = os.path.join(pt_image_path, 'negative_cigar_new_img.pt')
    ins_gold_folder = data_dir+'ins_gold'
    del_gold_folder = data_dir+'del_gold'

    # Load negative, ins, and del data
    negative_images = load_images_from_pt(negative_path)

    negative_images_number=len(negative_images)

    if negative_images_number<50:
        print(f"the number of the chromosome {chromosome} is few")
        return

    # Load ins and del gold images (from the folders)
    ins_gold_images = []
    del_gold_images = []
    for i in range(1, 51):  # 10 images for ins and del
        ins_gold_images.append(Image.open(os.path.join(ins_gold_folder, f"{i}.png")).convert('L'))
        del_gold_images.append(Image.open(os.path.join(del_gold_folder, f"{i}.png")).convert('L'))

    # Convert to tensors
    ins_gold_images = [np.array(img)for img in ins_gold_images]
    del_gold_images = [np.array(img) for img in del_gold_images]
    ins_gold_images = torch.tensor(ins_gold_images)
    del_gold_images = torch.tensor(del_gold_images)

    # Flatten and perform PCA on negative, ins_gold, and del_gold images
    negative_pca = pca_transform(negative_images, n_components=50)
    ins_gold_pca = pca_transform(ins_gold_images, n_components=50)
    del_gold_pca = pca_transform(del_gold_images, n_components=50)

    # K-means clustering on negative images PCA
    # 6-10个类是最佳选择，10
    cluster_num=10
    kmeans = KMeans(n_clusters=15,n_init=50)  # Adjust number of clusters as needed
    negative_clusters = kmeans.fit_predict(negative_pca)

    # Process each cluster
    selected_images = []
    to_remove_images = []

    ins_delete_one_cluster = 0
    del_delete_one_cluster = 0

    for cluster_id in range(kmeans.n_clusters):
        cluster_images_idx = np.where(negative_clusters == cluster_id)[0]
        cluster_images = negative_images[cluster_images_idx]
        cluster_pca = negative_pca[cluster_images_idx]
        print(f"cluster_id {cluster_id}:{len(cluster_pca)};",end='')
    print()

    for cluster_id in range(kmeans.n_clusters):
        cluster_images_idx = np.where(negative_clusters == cluster_id)[0]
        cluster_images = negative_images[cluster_images_idx]
        cluster_pca = negative_pca[cluster_images_idx]

        if len(cluster_pca)/negative_images_number>=0.2:#0.1或者0.2
            selected_images.extend(cluster_images)
            continue

        if len(cluster_pca)/negative_images_number<=0.05:#0.05
            to_remove_images.extend(cluster_images)
            continue

        # for probability between 0.05 and 0.5
        # Randomly pick one image from this cluster
        img_Ci = cluster_images[np.random.choice(len(cluster_images))]
        img_Ci_pca = cluster_pca[np.random.choice(len(cluster_images))]

        # e1 = 0.6  # Threshold 0.8
        # e2 = 1

        e1 = 0.6  # Threshold 0.8
        e2 = 1

        # Compare with ins_gold PCA
        num_ins=0
        for pca in ins_gold_pca:
            sim = compute_similarity(img_Ci_pca, pca)
            if sim >= e1:
                num_ins+=1

        if num_ins >= e2:
            if ins_delete_one_cluster==0:
                ins_delete_one_cluster=1
                # Remove all images in this cluster
                to_remove_images.extend(cluster_images)
        else:
            # Compare with del_gold PCA
            num_del = 0
            for pca in del_gold_pca:
                sim = compute_similarity(img_Ci_pca, pca)
                if sim >= e1:
                    num_del += 1

            if num_del >= e2:
                if del_delete_one_cluster==0:
                    del_delete_one_cluster=1
                    # Remove all images in this cluster
                    to_remove_images.extend(cluster_images)
            else:
                # Keep this cluster's images
                selected_images.extend(cluster_images)

    # Save selected images to negative_selected.pt
    if len(selected_images)>0:
        selected_images_tensor = torch.stack(selected_images).unsqueeze(1)
        torch.save(selected_images_tensor, pt_image_path+'/negative_cigar_new_img.pt')

    print(f"negative image number:{negative_images_number}")
    print(f"true negative image number:{len(selected_images)}")
    print(f"The number of deleted items:{negative_images_number-len(selected_images)}")
    print(f"selected probability:{(len(selected_images)/negative_images_number):.2f}")

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    chr_name_list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']
    for chr_name in chr_name_list:
        process_images(chr_name)