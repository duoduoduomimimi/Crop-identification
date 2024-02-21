import  cv2
import os,shutil
import matplotlib.pyplot as plt
# 显示汉字用
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#找亮光位置
def create_mask(imgpath):
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return mask
#修复图片
def xiufu(imgpath,maskpath):
    src_ = cv2.imread(imgpath)
    mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    #缩放因子(fx,fy)
    res_ = cv2.resize(src_,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)#修复
    return dst

if __name__=='__main__':
    rootpath = ""
    masksavepath="mask"
    savepath = "result"
    path = 'chengzi.jpg'
    img_orig = cv2.imread(path, 1)
    maskpath =os.path.join(masksavepath, "mask_"+path)
    cv2.imwrite(maskpath, create_mask(path))
    dst=xiufu(path,maskpath)
    newname = 'xiufu_' + path.split(".")[0]
    cv2.imwrite(os.path.join(savepath, newname + ".jpg"), dst)
    plt.subplot(121), plt.title('原图像'), plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(122), plt.title('去除高光'), plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.show()