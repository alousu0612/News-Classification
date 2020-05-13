import os
import shutil
from os import listdir
from os.path import isfile, isdir, join

old_path = '../THUCNews/'
new_path = '../data/news/'

MaxCount = 13000

# category = os.listdir(data_path)
categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

print(categories)
for category in categories:
    dir = old_path + category
    # print(dir)
    new_dir = new_path + category
    data = os.listdir(dir)[:MaxCount]
    # print(len(data))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    for file in data:
        data_path = dir + '/' + file
        data_new = new_dir + '/' + file
        shutil.copyfile(data_path, data_new)
