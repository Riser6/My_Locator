#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import PIL.Image as Image
import os


ROOT = "./output"
classes = ['car', 'bird', 'turtle', 'dog', 'lizard']

IMAGES_FORMAT = ['.JPEG']  # 图片格式
IMAGE_SIZE = 600  # 每张小图片的大小
IMAGE_ROW = 6  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列

# 获取图片集地址下的所有图片名称


def image_compose(IMAGES_PATH, image_names, IMAGE_SAVE_PATH):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + "/" + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    return to_image.save(IMAGE_SAVE_PATH)

for cls in classes:
    IMAGES_PATH = os.path.join(ROOT, cls)
    IMAGE_SAVE_PATH = os.path.join(ROOT, cls+".JPEG")
    image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

    print(cls)
    print(len(os.listdir(IMAGES_PATH)))
    print(len(image_names))
    if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")
    image_compose(IMAGES_PATH, image_names, IMAGE_SAVE_PATH)


