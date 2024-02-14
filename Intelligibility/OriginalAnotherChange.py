def remove_first_five_chars(filename, new_filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(new_filename, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line[4:])

# 使用方法
remove_first_five_chars('OriginalChange.txt', 'OriginalFinal.txt')
