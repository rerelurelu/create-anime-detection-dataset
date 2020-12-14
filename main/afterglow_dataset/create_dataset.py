from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection
from augmentation import name_list

names = name_list()
num_classes = len(names)


def create_data():
    X = []
    Y = []
    for index, name in enumerate(names):
        img_dir = name
        files = glob.glob(img_dir + '/*')
        for i, img in enumerate(files):
            image = Image.open(img)
            image = image.convert('RGB')
            data = np.asarray(image)
            X.append(data)
            Y.append(index)

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, y_train, y_test)
    np.save('dataset.npy', xy)


create_data()
