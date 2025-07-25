# FusionSVFilter：A Deep-learning based Fast Structural Variation Filtering Tool for Long Reads

## Introduction

Structural variations (SVs) play an important role in human diseases, biological evolution, and species diversity. The second-generation sequencing technologies are limited in detecting complex SVs due to their short read lengths. Although the third-generation sequencing technologies overcome this limitation by spanning complex genomic regions, the complexity of SVs, limitations of detection algorithms, and high single-base error rates still lead to many false positive (FP) calls. To address this challenge, we propose FusionSVFilter, a deep learning-based algorithm for SV filtering. FusionSVFilter applies a sequence-to-image transformation, converting sequence features, especially CIGAR strings in alignment results, into multi-level grayscale images. Moreover, the image-encoding process is accelerated with parallel computing. These grayscale images are then enhanced and transformed into RGB format. Additionally, we use principal component analysis (PCA) and K-means clustering to clean the training data by removing incorrectly labeled negative samples. To further improve performance, FusionSVFilter leverages the pre-trained ResNet50 model, utilizing transfer learning and fine-tuning on our dataset. Experimental results demonstrate that FusionSVFilter effectively reduces FP calls while maintaining the true positive (TP) calls almost unchanged.

## Experimental Configuration
Firstly, anaconda3 needs to be installed:
`Anaconda3-2024.02-1-Linux-x86_64.sh`
Download Anaconda 3-2024.02-1-Lux-x86_64.sh from the Anaconda official website and place it in a certain path.
`$ chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh`
`$ ./Anaconda3-2024.02-1-Linux-x86_64.sh`

Add conda to PATH:
`$ source ~/D/anaconda3/bin/activate`

Enable base environment:
`$ conda activate base`

create a new environment
`$ conda create -n fsv-filter python=3.6 -y`

activate the fsv-filter environment
`$ conda activate fsv-filter`

install pytorch/cudatoolkit/torchvision/torchaudio:
`$ conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit==11.3.1 -c pytorch -y`

install numpy:
`$ pip install numpy==1.19.2`

install pytorch-lightning:
`$ conda install pytorch-lightning=1.5.10 -c conda-forge -y`

install redis:
`$ conda install redis -y`

install scikit-learn:
`$ conda install scikit-learn -y`

install matplotlib:
`$ conda install matplotlib -y`

install samtools:
`$ conda install samtools -c bioconda`

install parallel:
`$ sudo apt install parallel` 

install ray[tune]:
`$ pip install ray[tune]==1.6.0`

install pudb:
`$ pip install pudb`

install hyperopt:
`$ pip install hyperopt`

install pysam:
`$ pip install pysam==0.15.4`

install pybind11:
`$ pip install pybind11`

### htslib (C++version of Samtools)
In this paper, we developed an image encoding parallel acceleration program in C++, so we need to install htslib and hdf5.
Two installation methods for htslib:
1. 
`$ sudo apt install libhts-dev`

2.
After installing Samtools, htslib was also installed

### hdf5(C++ & python)
C++:
```
$ sudo apt update
$ sudo apt install libhdf5-dev
```

python:
`$ pip install h5py`

### setup.py(the bridge file of Python calls C++)
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'chr_module', # 模块名
        
        ['chromosome.cpp'], # C++文件（图像编码并行加速程序）
        include_dirs=[
            pybind11.get_include(),
            '/home/zxh/anaconda3/envs/csv-filter/include',  # Conda 环境中 hdf5 的头文件路径
            '/home/zxh/anaconda3/envs/csv-filter/include/htslib',  # Conda 环境中 htslib 的头文件路径
        ],
        language='c++',
        extra_compile_args=[
            "-std=c++11",
            "-O2",
            "-mavx2",
            "-fopenmp",
        ],
        extra_link_args=[
            "-L/home/zxh/anaconda3/envs/csv-filter/lib",   # Conda 环境中的库文件路径
            "-lhts",                                       # 链接 htslib
            "-lhdf5",                                      # 链接 HDF5
        ],
    ),
]

setup(
    name='chr_module',
    ext_modules=ext_modules,
)

If you want to recompile chromosome.cpp, please use the following command：
`$ python setup.py clean --all`
`$ python setup.py build_ext --inplace`

If you encounter this problem:"absl-py version 2.0.0 is too new, Downgrade absl-py from version 2.0.0 to version 1.4.0":
`$ pip uninstall absl-py`
`$ pip install absl-py==1.4.0`

### conclusion
* python 3.6
* pytorch 1.10.2
* torchvision 1.10.2
* numpy 1.19.2
* pytorch-lightning 1.5.10
* redis 4.3.6
* scikit-learn 0.24.2
* samtools 1.5
* ray[tune] 1.6.0
* hyperopt 0.2.7
* pysam 0.15.4
* pybind11 2.13.6
* gcc 9.4.0
* g++ 9.4.0

## dataset
### HG002
Tier1 benchmark SV callset and high-confidence HG002 region: [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/analysis/NIST_SVs_Integration_v0.6/)
PacBio 70x (CLR): [https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/](https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/PacBio_MtSinai_NIST/)
Oxford Nanopore ultralong (guppy-V3.2.4\_2020-01-22): [ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz](ftp://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/data/AshkenazimTrio/HG002_NA24385_son/Ultralong_OxfordNanopore/guppy-V3.2.4_2020-01-22/HG002_ONT-UL_GIAB_20200122.fastq.gz)

### reference
GRCh37: [https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz](https://ftp.ensembl.org/pub/release-75/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.75.dna.primary_assembly.fa.gz)

##model
ResNet50_2C_97.ckpt(https://github.com/nudt-bioinfo/FusionSVFilter/tree/main/FusionSVFilter-main_2C/models-self)

## usage

### train model
In the src folder

preprocess VCF data:
`$ python vcf_data_process.py`

Generate grayscale images using a parallel image encoding program compiled with C++:
`$ python parallel_image_encoding_acceleration_file.py`

Clean Negative sample in the dataset:
`$ python clean_dataset.py`

Merge image data and label:
`$ python data_spread.py`

train:
`$ python train.py`

### predict and filter
predict:
`$ python predict.py`

filter:
`$ python filter_predict.py`

## Expansion: structural variation fusion filtering
e.g. For VCF files detected by PBSV(pbsv.vcf) and cuteSV(cutesv.vcf):
`$ python merge_vcf_files_imprecise_without_ID.py`

and so forth

Finally obtaining cutesv_pbsv_sniffles_and_so_on.vcf

fusion filtering

## Contact
For advising, bug reporting and requiring help, please contact [yingbocui@nudt.edu.cn](yingbocui@nudt.edu.cn).



