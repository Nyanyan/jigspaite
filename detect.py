import cv2
import numpy as np
from PIL import Image, ImageDraw
from functools import reduce

def scale_box(img, width, height):
    h, w = img.shape[:2]
    aspect = w / h
    if width / height >= aspect:
        nh = height
        nw = round(nh * aspect)
    else:
        nw = width
        nh = round(nw / aspect)
    dst = cv2.resize(img, dsize=(nw, nh))
    return dst

def find_contours(name):
    im = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    _, im_bw = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow('mask', im_bw)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(im_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    res = []
    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        res.append(contour)
    return res

filename = 'img.jpg'

image = Image.open(filename)
contours = find_contours(filename)

for i, contour in enumerate(contours):
    # 輪郭データからpolygonに渡すための変換（もっと良い方法があるかもしれない）
    area = tuple(map(lambda c: tuple(c[0]), contour.tolist()))
    # マスク画像を作成
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(area, fill=255)
    copy = image.copy()
    copy.putalpha(mask)
    # 切り抜き
    bbox = reduce(lambda p, c: [
            min(p[0], c[0]), min(p[1], c[1]),
            max(p[2], c[0]), max(p[3], c[1])
        ], area, [image.width, image.height, 0, 0])
    output = copy.crop(bbox)
    output.save('shape_' + str(i+1) + '.png')

'''

img = cv2.imread('img.jpg')
img = scale_box(img, 500, 500)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)

print(type(mask))
cv2.imshow('mask', mask)
cv2.waitKey(0)
'''