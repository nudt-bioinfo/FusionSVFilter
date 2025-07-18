# import subprocess
# import os
# import multiprocessing
# from utilities import mymkdir
# import time
#
# bam_start_time=time.time()
#
# # 设置 BAM 文件和输出路径
# bam_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
# bam_name = "HG002-PacBio-HiFi-minimap2.sorted.bam"
#
# # bam_dir = "/mnt/HHD_16T_1/Alignment_data/NA12878/"
# # bam_name = "sorted_final_merged.bam"
#
# data_dir = "../data/"
# # data_dir = "/mnt/HHD_16T_1/F-SV/HG002_data/"
#
# depth_dir = os.path.join(data_dir, "depth/")
#
# # 指定您希望使用的总线程数量
# total_threads = 24  # 您指定的总线程数量
#
# # 目标 Socket ID（0 或 1）
# target_socket_id = 0  # 绑定到 Socket 0
#
# # 获取指定 Socket 上的 CPU 核心列表
# def get_cpus_for_socket(target_socket_id):
#     cpus_for_socket = []
#     cpu_dir = "/sys/devices/system/cpu/"
#     for cpu in os.listdir(cpu_dir):
#         if cpu.startswith("cpu") and cpu[3:].isdigit():
#             cpu_id = int(cpu[3:])
#             topology_path = os.path.join(cpu_dir, cpu, "topology", "physical_package_id")
#             if os.path.exists(topology_path):
#                 with open(topology_path, 'r') as f:
#                     socket_id = int(f.read().strip())
#                 if socket_id == target_socket_id:
#                     cpus_for_socket.append(cpu_id)
#     return cpus_for_socket
#
# # 获取目标 Socket 的 CPU 列表
# cpus_for_socket = get_cpus_for_socket(target_socket_id)
# if len(cpus_for_socket) < total_threads:
#     print(f"> warning: Socket {target_socket_id} CPU number less than total thread number.")
#     total_threads = len(cpus_for_socket)
#
# # 设置主进程的 CPU 亲和性
# try:
#     os.sched_setaffinity(0, cpus_for_socket)
#     print(f"> set CPU affinity to Cores: {cpus_for_socket}")
# except AttributeError:
#     print("> os.sched_setaffinity cannot be used on this system.")
# except Exception as e:
#     print(f"> set CPU affinity failed: {e}")
#
# # 创建 depth 目录（如果不存在）
# mymkdir(depth_dir)
#
# # 获取 BAM 文件中的所有染色体名称
# def get_chromosome_list(bam_file):
#     cmd = f"samtools idxstats {bam_file}"
#     result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#     chromosomes = []
#     for line in result.stdout.strip().split('\n'):
#         fields = line.split('\t')
#         if len(fields) >= 1 and fields[0] != '*' and int(fields[2]) > 0:
#             chromosomes.append(fields[0])
#     return chromosomes
#
# bam_file_path = os.path.join(bam_dir, bam_name)
# chromosomes = get_chromosome_list(bam_file_path)
#
# # 定义处理单个染色体的函数
# def process_chromosome(chromosome):
#     try:
#         # 设置子进程的 CPU 亲和性
#         try:
#             os.sched_setaffinity(0, cpus_for_socket)
#         except AttributeError:
#             pass
#         except Exception as e:
#             print(f"> Subprocess set CPU affinity failed: {e}")
#
#         output_file = os.path.join(depth_dir, chromosome)
#         if os.path.exists(output_file):
#             print(f"> File {chromosome} exists, skipping.")
#             return
#
#         print(f"> Processing chromosome {chromosome}...")
#         cmd = f"samtools depth -r {chromosome} -@ 1 {bam_file_path} > {output_file}"
#         subprocess.call(cmd, shell=True)
#         print(f"> Finished chromosome {chromosome}")
#     except Exception as e:
#         print(f"> Error processing chromosome {chromosome}: {e}")
#
# # 确定进程池大小
# pool_size = min(len(chromosomes), total_threads)
# print(f"> Using a pool size of {pool_size} with total threads {total_threads}")
#
# # 创建进程池
# pool = multiprocessing.Pool(processes=pool_size)
#
# # 提交任务到进程池
# results = []
# for chrom in chromosomes:
#     result = pool.apply_async(process_chromosome, args=(chrom,))
#     results.append(result)
#
# # 关闭进程池，不再接受新任务
# pool.close()
#
# # 等待所有进程完成
# pool.join()
#
# # 检查是否有异常
# for result in results:
#     try:
#         result.get()
#     except Exception as e:
#         print(f"> Exception occurred during processing: {e}")
#
# print("====== All chromosomes processed ======")
# bam_end_time=time.time()
# bam_time=(bam_end_time-bam_start_time)*1000
# print(f"the time of processing *.bam:{bam_time:.3f}ms")


# import subprocess
# import os
# import multiprocessing
# from utilities import mymkdir
# import time
#
# bam_start_time = time.time()
#
# # 设置 BAM 文件和输出路径
# bam_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
# bam_name = "HG002-PacBio-HiFi-minimap2.sorted.bam"
#
# # bam_dir = "/mnt/HHD_16T_1/Alignment_data/NA12878/"
# # bam_name = "sorted_final_merged.bam"
# data_dir = "../data/"
#
# # data_dir = "/mnt/HHD_16T_1/F-SV/HG002_data/"
# depth_dir = os.path.join(data_dir, "depth/")
#
# # 指定您希望使用的总线程数量
# total_threads = 24  # 您指定的总线程数量
#
# # 目标 Socket ID（0 或 1）
# target_socket_id = 0  # 绑定到 Socket 0
#
# # 获取指定 Socket 上的 CPU 核心列表
# def get_cpus_for_socket(target_socket_id):
#     cpus_for_socket = []
#     cpu_dir = "/sys/devices/system/cpu/"
#     for cpu in os.listdir(cpu_dir):
#         if cpu.startswith("cpu") and cpu[3:].isdigit():
#             cpu_id = int(cpu[3:])
#             topology_path = os.path.join(cpu_dir, cpu, "topology", "physical_package_id")
#             if os.path.exists(topology_path):
#                 with open(topology_path, 'r') as f:
#                     socket_id = int(f.read().strip())
#                     if socket_id == target_socket_id:
#                         cpus_for_socket.append(cpu_id)
#     return cpus_for_socket
#
# # 获取目标 Socket 的 CPU 列表
# cpus_for_socket = get_cpus_for_socket(target_socket_id)
# if len(cpus_for_socket) < total_threads:
#     print(f"> warning: Socket {target_socket_id} CPU number less than total thread number.")
#     total_threads = len(cpus_for_socket)
#
# # 设置主进程的 CPU 亲和性
# try:
#     os.sched_setaffinity(0, cpus_for_socket)
#     print(f"> set CPU affinity to Cores: {cpus_for_socket}")
# except AttributeError:
#     print("> os.sched_setaffinity cannot be used on this system.")
# except Exception as e:
#     print(f"> set CPU affinity failed: {e}")
#
# # 创建 depth 目录（如果不存在）
# mymkdir(depth_dir)
#
# # 获取 BAM 文件中的所有染色体名称
# def get_chromosome_list(bam_file):
#     cmd = f"samtools idxstats {bam_file}"
#     result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
#     chromosomes = []
#     for line in result.stdout.strip().split('\n'):
#         fields = line.split('\t')
#         if len(fields) >= 1 and fields[0] != '*' and int(fields[2]) > 0:
#             chromosomes.append(fields[0])
#     return chromosomes
#
# bam_file_path = os.path.join(bam_dir, bam_name)
# chromosomes = get_chromosome_list(bam_file_path)
#
# # 定义处理单个染色体的函数
# def process_chromosome(chromosome):
#     try:
#         # 清除系统缓存
#         subprocess.call("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
#
#         # 设置子进程的 CPU 亲和性
#         try:
#             os.sched_setaffinity(0, cpus_for_socket)
#         except AttributeError:
#             pass
#         except Exception as e:
#             print(f"> Subprocess set CPU affinity failed: {e}")
#
#         output_file = os.path.join(depth_dir, chromosome)
#         if os.path.exists(output_file):
#             print(f"> File {chromosome} exists, skipping.")
#             return
#
#         print(f"> Processing chromosome {chromosome}...")
#         cmd = f"samtools depth -r {chromosome} -@ 1 {bam_file_path} > {output_file}"
#         subprocess.call(cmd, shell=True)
#         print(f"> Finished chromosome {chromosome}")
#     except Exception as e:
#         print(f"> Error processing chromosome {chromosome}: {e}")
#
# # 确定进程池大小
# pool_size = min(len(chromosomes), total_threads)
# print(f"> Using a pool size of {pool_size} with total threads {total_threads}")
#
# # 创建进程池
# pool = multiprocessing.Pool(processes=pool_size)
#
# # 提交任务到进程池
# results = []
# for chrom in chromosomes:
#     result = pool.apply_async(process_chromosome, args=(chrom,))
#     results.append(result)
#
# # 关闭进程池，不再接受新任务
# pool.close()
#
# # 等待所有进程完成
# pool.join()
#
# # 检查是否有异常
# for result in results:
#     try:
#         result.get()
#     except Exception as e:
#         print(f"> Exception occurred during processing: {e}")
#
# print("====== All chromosomes processed ======")
#
# bam_end_time = time.time()
# bam_time = (bam_end_time - bam_start_time) * 1000
# print(f"the time of processing *.bam:{bam_time:.3f}ms")






import subprocess
import os
import multiprocessing
from utilities import mymkdir
import time

bam_start_time = time.time()

# 设置 BAM 文件和输出路径
bam_dir = "/mnt/HHD_16T_1/Alignment_data/HG002/PacBio-HiFi/"
bam_name = "HG002-PacBio-HiFi-minimap2.sorted.bam"

# bam_dir = "/mnt/HHD_16T_1/Alignment_data/NA12878/"
# bam_name = "sorted_final_merged.bam"
data_dir = "../data/"

# data_dir = "/mnt/HHD_16T_1/F-SV/HG002_data/"
depth_dir = os.path.join(data_dir, "depth/")

# 指定您希望使用的总线程数量
total_threads = 24  # 您指定的总线程数量

# 目标 Socket ID（0 或 1）
target_socket_id = 0  # 绑定到 Socket 0

# 获取指定 Socket 上的 CPU 核心列表
def get_cpus_for_socket(target_socket_id):
    cpus_for_socket = []
    cpu_dir = "/sys/devices/system/cpu/"
    for cpu in os.listdir(cpu_dir):
        if cpu.startswith("cpu") and cpu[3:].isdigit():
            cpu_id = int(cpu[3:])
            topology_path = os.path.join(cpu_dir, cpu, "topology", "physical_package_id")
            if os.path.exists(topology_path):
                with open(topology_path, 'r') as f:
                    socket_id = int(f.read().strip())
                    if socket_id == target_socket_id:
                        cpus_for_socket.append(cpu_id)
    return cpus_for_socket

# 获取目标 Socket 的 CPU 列表
cpus_for_socket = get_cpus_for_socket(target_socket_id)
if len(cpus_for_socket) < total_threads:
    print(f"> warning: Socket {target_socket_id} CPU number less than total thread number.")
    total_threads = len(cpus_for_socket)

# 设置主进程的 CPU 亲和性
try:
    os.sched_setaffinity(0, cpus_for_socket)
    print(f"> set CPU affinity to Cores: {cpus_for_socket}")
except AttributeError:
    print("> os.sched_setaffinity cannot be used on this system.")
except Exception as e:
    print(f"> set CPU affinity failed: {e}")

# 创建 depth 目录（如果不存在）
mymkdir(depth_dir)

# 获取 BAM 文件中的所有染色体名称
def get_chromosome_list(bam_file):
    cmd = f"samtools idxstats {bam_file}"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    chromosomes = []
    for line in result.stdout.strip().split('\n'):
        fields = line.split('\t')
        if len(fields) >= 1 and fields[0] != '*' and int(fields[2]) > 0:
            chromosomes.append(fields[0])
    return chromosomes

bam_file_path = os.path.join(bam_dir, bam_name)
chromosomes = get_chromosome_list(bam_file_path)

# 定义处理单个染色体的函数
def process_chromosome(chromosome):
    try:
        # 设置子进程的 CPU 亲和性
        try:
            os.sched_setaffinity(0, cpus_for_socket)
        except AttributeError:
            pass
        except Exception as e:
            print(f"> Subprocess set CPU affinity failed: {e}")

        output_file = os.path.join(depth_dir, chromosome)
        if os.path.exists(output_file):
            print(f"> File {chromosome} exists, skipping.")
            return

        print(f"> Processing chromosome {chromosome}...")
        cmd = f"samtools depth -r {chromosome} -@ 1 {bam_file_path} > {output_file}"
        subprocess.call(cmd, shell=True)
        print(f"> Finished chromosome {chromosome}")
    except Exception as e:
        print(f"> Error processing chromosome {chromosome}: {e}")

# 确定进程池大小
pool_size = min(len(chromosomes), total_threads)
print(f"> Using a pool size of {pool_size} with total threads {total_threads}")

# 创建进程池
pool = multiprocessing.Pool(processes=pool_size)

# 提交任务到进程池
results = []
for chrom in chromosomes:
    result = pool.apply_async(process_chromosome, args=(chrom,))
    results.append(result)

# 关闭进程池，不再接受新任务
pool.close()

# 等待所有进程完成
pool.join()

# 检查是否有异常
for result in results:
    try:
        result.get()
    except Exception as e:
        print(f"> Exception occurred during processing: {e}")

# 打印总时间
bam_end_time = time.time()
bam_time = (bam_end_time - bam_start_time) * 1000
print(f"the time of processing *.bam:{bam_time:.3f}ms")

# 确保清除文件系统缓存（为了避免缓存带来的性能差异）
def clear_cache():
    try:
        subprocess.call("sync", shell=True)  # 强制写入磁盘
        subprocess.call("echo 3 > /proc/sys/vm/drop_caches", shell=True)  # 清除文件系统缓存
        print("> Cleared system cache.")
    except Exception as e:
        print(f"> Failed to clear cache: {e}")

# clear_cache()

print("====== All chromosomes processed ======")