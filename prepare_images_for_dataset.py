from imutils import paths
from PIL import Image, ImageDraw
import os
import glob
import cv2 as cv
import numpy as np


for file in glob.glob("data/5/*.png", recursive=True):
    img = Image.open(file)
    new_img = Image.new("RGBA", img.size, "WHITE")
    new_img.paste(img, (0, 0), img)
    new_img.convert('RGB').save(file.replace("png", "jpg"), "JPEG")
    img = cv.imread(file.replace("png", "jpg"))
    img = np.array(img)
    img = 255 - img
    cv.imwrite(file.replace("png", "jpg"), img)
    os.remove(file)

for file in glob.glob("data/5/*.jpg", recursive=True):
    image = cv.imread(file)
    image = cv.resize(image, (8, 8), interpolation=cv.INTER_AREA)
    img = np.array(image)
    img = 255 - img
    cv.imwrite(file, img)
