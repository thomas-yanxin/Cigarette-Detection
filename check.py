import sys
import cv2
for line in sys.stdin:
    items = line.strip().split()
    try:
        im = cv2.imread(items[0])
        im = im.astype('float32')
    except:
        print("Wrong img file:", items[0])