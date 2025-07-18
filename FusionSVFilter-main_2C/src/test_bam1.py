import pysam

def print_bam_header(file_path):
    # 打开 BAM 文件
    with pysam.AlignmentFile(file_path, "rb") as bam_file:
        # 打印 BAM 文件的头部信息（通常包含 6 行或更多信息）
        print("BAM Header:")
        for line in bam_file.header['HD']:
            print(line)

def print_first_n_reads(file_path, n=10,m=12):
    # 打开 BAM 文件
    with pysam.AlignmentFile(file_path, "rb") as bam_file:
        # 读取前 n 行（前 n 条比对记录）
        print(f"First {n} reads:")
        for i, read in enumerate(bam_file.fetch()):
            if i>=n and i<=m:
                print(read)
                print(f"read.cigar:{read.cigar}")
            elif i>m:
                break

# 指定 BAM 文件的路径
bam_file_path = '../data/test.bam'

# 打印 BAM 文件头部信息
print_bam_header(bam_file_path)

# 打印前 10 条比对记录
print_first_n_reads(bam_file_path, n=100,m=102)

