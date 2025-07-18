import pysam

# 打开一个 BAM 文件
sam_file = pysam.AlignmentFile("../data/test.bam", "rb")

# 设置要提取的染色体和位置范围
chromosome = 10
begin = 3689814
end = 3691862

# 使用 fetch 提取指定区域的读取
for read in sam_file.fetch(chromosome):
    print(read)

# 关闭文件
sam_file.close()

