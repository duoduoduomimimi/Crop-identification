本项目采用了深度学习ResNet50的网络结构用于模型训练，使用了限制对比度自适应直方图均衡化（CLAHE）、以及openCV库中的inpaint（修复函数）、threshol（阈值处理寻找高光区函数）等图像处理方法对原始数据集进行预处理，
使农作物图片的图像细节更加完整，同时能够有效解决图像光照不均匀的问题，以来提高模型识别的准确率。
