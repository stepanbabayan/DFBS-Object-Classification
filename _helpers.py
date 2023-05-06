import numpy as np
import pandas as pd
from PIL import Image

from os import mkdir
from shutil import rmtree


def loadImages(X):
    images_list = []

    for i, path in enumerate(X):
        im = Image.open(path)
        arr = np.array(im)

        arr = (arr-arr.min())/(arr.max()-arr.min())

        images_list.append(arr)

    return np.array(images_list)


def prepareData(path):
    data = pd.read_csv(path)
    if 'Unnamed: 0' in data:
        data.set_index('Unnamed: 0', inplace=True)
    return data


def make_directory(path):
    folders = path.split('/')
    current_path = ''
    for folder in folders[:-1]:
        current_path += folder + '/'
        try:
            mkdir(current_path)
        except OSError:
            pass

    current_path += folders[-1]
    rmtree(current_path, ignore_errors=True)
    mkdir(current_path)
