'''
Interface with ia870 toolbox (https://github.com/MICLab-Unicamp/ia870)
'''

import numpy as np
from PIL import Image
from ia870 import ialabel
from utils import show_grayimage, ESC
import cv2 as cv
import time


def get_largest_components(volume, mask_ths=0.5, ncomponents=2, debug=False):
    '''
    Returns 2 or 1 largest components, input should be numpy in cpu
    '''
    begin = time.time()

    volume = volume > mask_ths

    if volume.astype(np.int).sum() == 0:
        return volume.astype(np.float32)

    labels = ialabel(volume, np.ones((3, 3, 3), dtype=bool))

    bincount = np.bincount(labels.flat)
    if debug:
        print("label count: {}".format(bincount))

    # Increasing sort (0, 1, 2 ,3...)
    sorted_count = np.sort(bincount)

    # Gets two labels in bincount with more area, excluding the object with higher area (background)
    largest_components = {}

    upper_limit = 4
    if len(bincount) == 2:
        upper_limit = 3

    for i in range(2, upper_limit):
        label = np.where(bincount == sorted_count[-i])[0][0]
        largest_components[str(label)] = sorted_count[-i]

    if debug:
        print("two highest labels excluding highest {}".format(largest_components))
        print("largest components processing time: {}s".format(time.time() - begin))

    highest_labels = np.asarray(list(largest_components.keys()), dtype=np.int64)
    if debug:
        print("Largest object labels: {}".format(highest_labels))

    # print(labels.max())
    # print(labels.min())
    # print(labels.dtype)
    hlabel_len = len(highest_labels)
    if hlabel_len >= 2:
        return (np.ma.masked_not_equal(labels, highest_labels[0]).filled(0) + np.ma.masked_not_equal(labels,
                highest_labels[1]).filled(0)).astype(np.bool).astype(np.float32)
    else:
        return np.ma.masked_not_equal(labels, highest_labels[0]).filled(0).astype(np.bool).astype(np.float32)


def ialabel_test():
    '''
    Tests labeling in sample 2D and 3D numpy
    '''

    # 2D numpy array from 0 to 24
    f = np.arange(24).reshape(4, 6) > 14

    # Label the 2D image, works!
    ialabel(f)

    # Label barcode
    fbarcode = Image.open('data/barcode.tif')
    show_grayimage("Barcode image", fbarcode)

    # Threhsold to binary image
    fbarc = np.array(fbarcode) < 128
    show_grayimage("Threshold barcode image", fbarc)

    # Label components of binary barcode image
    flabel = ialabel(fbarc)
    print("How many components in 2D test: {}".format(flabel.max()))
    show_grayimage("Labeled 2D image", flabel.astype(np.uint8))

    h = np.bincount(flabel.flat)
    print("Count for each label {}".format(h))

    print("Biggest component index {}".format(np.argmax(h[1:])))

    show_grayimage("Biggest component", (flabel == 340).astype(np.uint8)*255)

    print("3D test running in random 181, 217, 181 shape")

    f3 = np.zeros((181, 217, 181))
    cubesize = 20
    ncubes = 180//cubesize - 1
    for i in range(0, ncubes, 2):  # 2 step to give some empty space
        cubeshape = (cubesize + i, cubesize + i, cubesize + i)
        print("Inserting cube {} with shape {}".format(i//2, cubeshape))
        cube = np.ones(cubeshape)
        j = i*cubesize
        f3[j:j+cubeshape[0], j:j+cubeshape[0], j:j+cubeshape[0]] = cube

    for i in range(217):
        cv.imshow("before", f3[:, i, :])
        cv.waitKey(10)

    f3 = get_largest_components(f3)
    print(f3.dtype)
    for i in range(217):
        cv.imshow("after", f3[:, i, :])
        cv.waitKey(10)

    quit()
    # plt.show()

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

    while True:
        _, img = cap.read()
        # gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gimg = img
        timg = gimg > 128
        labeledint = ialabel(timg, np.ones((3, 3, 3), dtype=bool))
        labeled = labeledint.astype(np.float32)
        print("nlabels: {}".format(labeled.max()))

        labeled = (labeled - labeled.min()) / (labeled.max() - labeled.min())
        gimg = (gimg - gimg.min()) / (gimg.max() - gimg.min())
        timg = timg.astype(np.float32)

        biggest_labels = np.bincount(labeledint.flat)
        print(biggest_labels)
        biggest_label = np.argmax(biggest_labels)

        cv.imshow("5 Biggest Components", cv.resize(np.hstack((gimg, timg, labeled,
                                                               (labeledint == biggest_label).astype(np.float32))), (1920, 600)))
        if cv.waitKey(1) == ESC:
            quit()


if __name__ == "__main__":
    ialabel_test()
