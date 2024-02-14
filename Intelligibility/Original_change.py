def remove_lines(filename, new_filename):
    with open(filename, 'r',encoding='utf-8') as f:
        lines = f.readlines()

    with open(new_filename, 'w',encoding='utf-8') as f:
        for i in range(len(lines)):
            if (i + 1) % 3 != 0:
                f.write(lines[i])

# 使用方法
remove_lines('Original.txt', 'OriginalChange.txt')
