# import numpy as np
import os
import random
import shutil

root_dir = r'D:\onko\定位裁切图片\processedImg'  # 原始数据集的路径

exist_data = os.listdir(os.path.join(root_dir,'exist'))  # 原始数据集的图片
notExist_data = os.listdir(os.path.join(root_dir,'notExist'))  # 原始数据集的图片

exist_train_path=os.path.join(root_dir,'train','exist')  # 训练集的路径
exist_val_path=os.path.join(root_dir,'val','exist')  # 验证集的路径
notExist_train_path=os.path.join(root_dir,'train','notExist')  # 训练集的路径
notExist_val_path=os.path.join(root_dir,'val','notExist')  # 验证集的路径

def datasetImage(path, path1, path2, path3, path4):
    for filename in os.listdir(path):
        if not os.path.exists(path1):
            os.makedirs(path1)
        if not os.path.exists(path2):
            os.makedirs(path2)
        if not os.path.exists(path3):
            os.makedirs(path3)
        if not os.path.exists(path4):
            os.makedirs(path4)
    
    t = int(len(exist_data) * 0.7)
    for i in range(len(exist_data)):
        random.shuffle(exist_data)  # 打乱数据
    for z in range(len(exist_data)):  # 将数据按8：2分到train和test中
        print('z:', z, '\n')
        pic_path = os.path.join(path,'exist',exist_data[z])
        print('pic_path:', pic_path)
        if z < t:
            obj_path = os.path.join(path1,exist_data[z])
            shutil.copyfile(pic_path, obj_path)
            print('train:', obj_path)
        else:
            obj_path = os.path.join(path3,exist_data[z])
            print('test:', obj_path)  # 显示分类情况
            shutil.copyfile(pic_path, obj_path)  # 往train、val中复制图片
            
    t2=int(len(notExist_data)*0.7)
    for i in range(len(notExist_data)):
        random.shuffle(notExist_data)  # 打乱数据
    for z in range(len(notExist_data)):  # 将数据按8：2分到train和test中
        print('z:', z, '\n')
        pic_path = os.path.join(path,'notExist',notExist_data[z])
        print('pic_path:', pic_path)
        if z < t2:
            obj_path = os.path.join(path2,notExist_data[z])
            shutil.copyfile(pic_path, obj_path)
            print('train:', obj_path)
        else:
            obj_path = os.path.join(path4,notExist_data[z])
            print('test:', obj_path)  # 显示分类情况
            shutil.copyfile(pic_path, obj_path)  # 往train、val中复制图片

if __name__ == '__main__':
    datasetImage(root_dir, exist_train_path, notExist_train_path, exist_val_path,  notExist_val_path)

