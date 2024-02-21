import os
import glob
import random
import shutil
from PIL import  Image
# 数据集划分
if __name__ == '__main__':
    test_split_ratio = 0.05
    # 图片缩放统一大小
    desired_size =512
    raw_path = './农作物图像'
    dirs = glob.glob(os.path.join(raw_path,'*'))
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f'Rotally {len(dirs)} classes: {dirs}')
    # 对每一个类别进行处理
    lent=0
    for path in dirs:
        path = path.split('/')[-1]
        files = glob.glob(os.path.join(f'农作物图像/{path}', '*.jpg'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.JPG'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.png'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.PNG'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.jpeg'))
        lent=lent+len(files)
    print(lent)
    for path in dirs:
        path = path.split('/')[-1]
        os.makedirs(f'train/{path}',exist_ok=True)
        os.makedirs(f'test/{path}', exist_ok=True)
        files = glob.glob(os.path.join(f'农作物图像/{path}', '*.jpg'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.JPG'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.png'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.PNG'))
        files += glob.glob(os.path.join(f'农作物图像/{path}', '*.jpeg'))

        random.shuffle(files)
        # 训练集和测试集的边界
        boundary = int(len(files)*test_split_ratio)
        for i,file in enumerate(files):
            # 将图片转换为RGB
            img = Image.open(file).convert('RGB')
            old_size = img.size
            ratio = float(desired_size)/max(old_size)
            # 将图片缩放到目标的大小（等比例缩放）
            new_size = tuple([int(x*ratio) for x in old_size])
            # 高质量调整图片大小
            im = img.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im,((desired_size-new_size[0])//2,(desired_size-new_size[1])//2))
            assert new_im.mode == 'RGB'
            if i <= boundary:
                new_im.save(os.path.join(f'test/{path}',file.split('/')[-1].split('.')[0]+'.jpg'))
            else:
                new_im.save(os.path.join(f'train/{path}',file.split('/')[-1].split('.')[0]+'.jpg'))#.split('.')[0] + '.jpg'
        print(path)
    test_files = glob.glob(os.path.join('test','*','*.jpg'))
    train_files = glob.glob(os.path.join('train','*', '*.jpg'))
    print(f'Totally{len(train_files)} files for training')
    print(f'Totally{len(test_files)} files for test')

