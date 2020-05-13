import os


def _read_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read().replace('\n', '').replace('\t', '').replace('\u3000', '')


def save_file(dirname):

    f_train = open('../data/cnews/cnews_train.txt', 'w', encoding='utf-8')
    f_test = open('../data/cnews/cnews_test.txt', 'w', encoding='utf-8')
    f_val = open('../data/cnews/cnews_val.txt', 'w', encoding='utf-8')
    for category in os.listdir(dirname):
        cat_dir = os.path.join(dirname, category)
        if not os.path.isdir(cat_dir):
            continue
        files = os.listdir(cat_dir)
        count = 0
        for cur_file in files:
            filename = os.path.join(cat_dir, cur_file)
            content = _read_file(filename)
            if count < 5000:
                f_train.write(category + '\t' + content + '\n')
            elif count < 6000:
                f_test.write(category + '\t' + content + '\n')
            else:
                f_val.write(category + '\t' + content + '\n')
            count += 1

        print('Finished:', category)

    f_train.close()
    f_test.close()
    f_val.close()


if __name__ == '__main__':
    save_file('../data/news')
    print(len(open('../data/cnews/cnews_train.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../data/cnews/cnews_test.txt', 'r', encoding='utf-8').readlines()))
    print(len(open('../data/cnews/cnews_val.txt', 'r', encoding='utf-8').readlines()))
