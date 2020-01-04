from PIL import Image
import os.path
import os


# 读取文件夹里的图片并
#resize
#保存到新文件夹
old_image_path = './data/train_data/'
new_image_path = './data/smaller_train_data/'

for img_name in os.listdir(old_image_path):
    img = Image.open(os.path.join(old_image_path,img_name))
    new_image = img.resize((512,512))
    new_image.save(os.path.join(new_image_path,img_name))
