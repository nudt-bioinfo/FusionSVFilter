# from setuptools import setup, Extension
# import pybind11
#
# ext_modules = [
#     Extension(
#         'chr_module',
#         ['chromosome.cpp'],
#         include_dirs=[pybind11.get_include()],
#         language='c++',
#         extra_compile_args=["-std=c++11", "-O2", "-mavx2", "-fopenmp", "-I/home/zxh/D/envs/include/htslib", "-I/home/zxh/D/envs/include/hdf5/serial"],
#         extra_link_args=["-L/home/zxh/D/envs/include/htslib", "-L/home/zxh/D/envs/lib/hdf5/serial", "-lhts", "-lhdf5"],
#     ),
# ]
#
# setup(
#     name='chr_module',
#     ext_modules=ext_modules,
# )

from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'chr_module',
        ['chromosome.cpp'],
        include_dirs=[
            pybind11.get_include(),
            '/home/username/anaconda3/envs/fsv-filter/include',  # Conda 环境中 hdf5 的头文件路径
            '/home/username/anaconda3/envs/fsv-filter/include/htslib',  # Conda 环境中 htslib 的头文件路径
        ],
        language='c++',
        extra_compile_args=[
            "-std=c++11",
            "-O2",
            "-mavx2",
            "-fopenmp",
        ],
        extra_link_args=[
            "-L/home/username/anaconda3/envs/fsv-filter/lib",   # Conda 环境中的库文件路径
            "-lhts",                                       # 链接 htslib
            "-lhdf5",                                      # 链接 HDF5
        ],
    ),
]

setup(
    name='chr_module',
    ext_modules=ext_modules,
)
