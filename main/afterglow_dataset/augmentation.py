import glob
import random
import cv2

random.seed(42)
data_dic = {'ran': 58, 'moca': 94, 'himari': 109, 'tsugumi': 111, 'tomoe': 87}


def name_list():
    names = [i for i in data_dic]
    return names


def augmentation():
    for lb, i in data_dic.items():
        path = lb + '/*.jpg'
        files = glob.glob(path)
        images = random.sample(files, k=i)
        counter = 1
        for img in images:
            tmp_img = cv2.imread(img)
            new_image = cv2.flip(tmp_img, 1)
            name = lb + '/new' + str(counter) + '.jpg'
            cv2.imwrite(name, new_image)
            counter += 1


augmentation()
