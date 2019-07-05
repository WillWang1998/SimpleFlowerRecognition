import cv2
import numpy as np
import random

flower_dict = {
    0: "daffodil",
    1: "snowdrop",
    2: "lily valley",
    3: "bluebell",
    4: "crocus",
    5: "iris",
    6: "tigerlily",
    7: "tulip",
    8: "fritillary",
    9: "sunflower",
    10: "daisy",
    11: "colt's foot",
    12: "dandelion",
    13: "cowslip",
    14: "buttercup",
    15: "windflower",
    16: "pansy"
}


def main():
    files_txt = open("./17flowers/jpg/files.txt", "r")
    files = files_txt.readlines()
    files_txt.close()
    for file in files:
        if file[-1] == '\n':
            file = file[:-1:]
        name = file
        name = name.replace("_", " ")
        name = name.replace(".", " ")
        name = name.split()
        number = int(name[1])
        print("Processing image %d/1360" % number)
        number -= 1
        number //= 80
        flower_name = flower_dict[number]
        image = cv2.imread("./17flowers/jpg/"+file)
        [h, w] = np.shape(image)[:-1]
        hw = min(h, w)
        sh = (h - hw) // 2
        sw = (w - hw) // 2
        file_prefix, file_suffix = file.split(".")
        image = image[sh:sh + hw, sw:sw + hw]
        image = cv2.resize(image, (299,299), cv2.INTER_NEAREST)
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_0." + file_suffix, image)
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_1." + file_suffix, cv2.flip(image, 0))
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_2." + file_suffix, cv2.flip(image, 1))
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_3." + file_suffix, cv2.flip(image, -1))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_h, image_s, image_v = cv2.split(image)
        h_max = np.max(image_h)
        s_max = np.max(image_s)
        v_max = np.max(image_v)
        image_random_1 = cv2.cvtColor(cv2.merge([(image_h * random.uniform(0, 255 / h_max)).astype(np.uint8),
                                                 (image_s * random.uniform(0, 255 / s_max)).astype(np.uint8),
                                                 (image_v * random.uniform(0, 255 / v_max)).astype(np.uint8)]),
                                      cv2.COLOR_HSV2BGR)
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_4." + file_suffix, image_random_1)
        image_random_2 = cv2.cvtColor(cv2.merge([(image_h * random.uniform(0, 255 / h_max)).astype(np.uint8),
                                                 (image_s * random.uniform(0, 255 / s_max)).astype(np.uint8),
                                                 (image_v * random.uniform(0, 255 / v_max)).astype(np.uint8)]),
                                      cv2.COLOR_HSV2BGR)
        cv2.imwrite("./flower_photos/" + flower_name + "/" + file_prefix + "_5." + file_suffix, image_random_2)


if __name__ == "__main__":
    main()
