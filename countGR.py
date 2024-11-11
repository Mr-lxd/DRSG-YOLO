import os


def count_germination(file_path):
    """
    统计单个文件的发芽数量和总样本数。
    :param file_path: 文件路径
    :return: 发芽数量, 总样本数
    """
    total_count = 0
    germinated_count = 0
    GR = 0

    # 打开文件并读取内容
    with open(file_path, 'r') as file:
        for line in file:
            data = line.split()
            # 只取第一个字段，代表发芽状态
            status = int(data[0])

            total_count += 1
            if status == 1:
                germinated_count += 1

        GR = germinated_count/total_count

    return germinated_count, total_count, GR


def process_all_files(directory_path):
    """
    遍历指定目录下的所有TXT文件，统计每个文件的发芽数量和总样本数。
    :param directory_path: TXT文件所在的目录路径
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            germinated_count, total_count, GR = count_germination(file_path)
            print(f"文件: {filename}, 发芽数量: {germinated_count}, 总样本数: {total_count}, 发芽率: {GR}")


# 设置文件目录路径
directory_path = "D:/DL_Project/data/test/labels"  # 替换为存放TXT文件的实际目录路径

# 处理所有文件
process_all_files(directory_path)
