
def split_txt_file(input_file, train_file, dev_file, test_file):
    # 读取原文件的所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 计算行数
    total_lines = len(lines)
    train_size = int(total_lines * 0.6)
    dev_size = int(total_lines * 0.2)

    # 切分数据
    train_lines = lines[:train_size]
    dev_lines = lines[train_size:train_size + dev_size]
    test_lines = lines[train_size + dev_size:]

    # 写入train.txt
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    # 写入dev.txt
    with open(dev_file, 'w', encoding='utf-8') as f:
        f.writelines(dev_lines)

    # 写入test.txt
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)


# 调用函数，进行拆分
split_txt_file('/Users/hwan/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/830da9def43de92ae65513342e7367b1/Message/MessageTemp/137f4c83ecf5057aec6d9038c0c6a9f9/File/产学研标注数据.txt',
               'data/train.txt', 'data/dev.txt', 'data/test.txt')


