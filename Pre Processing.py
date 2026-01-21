#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np

IMAGE_SIZE = 32


def list_folders(root_folder):
    """Function to get subdir list"""
    folder_list = []
    for folder in sorted(os.listdir(root_folder)):
        if os.path.isdir(os.path.join(root_folder, folder)):
            folder_list.append(folder)
    return folder_list


def create_folders(root_folder, folder_list):
    """Function to create folders in new dataset"""
    for folder in folder_list:
        os.makedirs(os.path.join(root_folder, folder), exist_ok=True)


def read_transparent_png(filename):
    """
    Change transparent bg to white
    """
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 0]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 10

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def clean(img):
    """Process an image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(__, img_bw) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    #ctrs, __ = cv2.findContours(img_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # take largest contour
    #ctr = sorted(ctrs, key=lambda ctr: (cv2.boundingRect(ctr)[2] * cv2.boundingRect(ctr)[3]),
                 #reverse=True)[0]
    # Get bounding box
    #x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    #roi = img_bw[y:y + h, x:x + w]
    return crop(gray, IMAGE_SIZE)


def crop(image, desired_size):
    """Crop and pad to req size"""
    #old_size = image.shape[:2]  # old_size is in (height, width) format
    #ratio = float(desired_size) / max(old_size)
    #new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(image, (32, 32))

    #delta_w = desired_size - new_size[1]
    #delta_h = desired_size - new_size[0]
    #top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    #left, right = delta_w // 2, delta_w - (delta_w // 2)

    #color = [0, 0, 0]
    #new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                #value=color)

    return im


def process_folder(folder):
    """Process all images in a folder"""
    extension = '.png'
    new_list = []
    for img in sorted(os.listdir(folder)):
        if img.endswith(extension):
            image = read_transparent_png(os.path.join(folder, img))
            new_img = clean(image)
            new_list.append([img, new_img])
            """except:
                print("\t" + img)"""
    return new_list


def save_new(folder, imglist):
    """Save newly created images"""
    for img in imglist:
        cv2.imwrite(os.path.join(folder, img[0]), img[1])


def process_images(raw_folder, clean_folder, folder_list):
    """Process the images"""
    for folder in folder_list:
        print(folder)
        imglist = process_folder(os.path.join(raw_folder, folder, 'output'))
        save_new(os.path.join(clean_folder, folder), imglist)


def skeletize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeroes = size - cv2.countNonZero(img)
        if zeroes == size:
            done = True

    return skel


# In[48]:


import cv2
import matplotlib.pyplot as plt
try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
words=[]
a=['banana', 'blue', 'book', 'boy', 'bush', 'child', 'flower', 'flowers', 'friday', 'fruit', 'fruits', 'girl', 'grapes', 'green', 'has', 'have', 'home', 'house', 'how', 'is', 'man', 'mango', 'monday', 'my', 'name', 'newspaper', 'our', 'paper', 'pen', 'people', 'pink', 'red', 'rose', 'saturday', 'sunday', 'that', 'their', 'there', 'this', 'thursday', 'tree', 'trees', 'tuesday', 'was', 'wednesday', 'were', 'when', 'why', 'woman', 'yellow']
for i in range(50):
    hindi=pytesseract.image_to_string(Image.open("D:\\project\\words\\"+a[i]+"\\"+a[i]+".png"),lang="mal")
    #print(hindi)
    
    words.append(hindi)

img=cv2.imread("D:\\project\\words\\blue\\blue.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hindi=pytesseract.image_to_string(Image.open("D:\\project\\words\\paper\\paper.png"),lang="mal")
print(hindi)

print(len(words))
print(words[23])


# In[21]:


import os
import shutil
import Augmentor


folder = "D:\\project\\try"
for f in list_folders(folder):
    if os.path.isdir(os.path.join(folder, f, 'output')):
        shutil.rmtree(os.path.join(folder, f, 'output'))
    p = Augmentor.Pipeline(os.path.join(folder, f))
    p.random_distortion(probability=1, grid_width=10, grid_height=10, magnitude=8)
    p.sample(200, multi_threaded=False)


# In[37]:


"""Module to clean the raw images"""



RAW_FOLDER = "D:\\project\\try"
CLEAN_FOLDER = "D:\\project\\try2"

FOLDER_LIST = list_folders(RAW_FOLDER)
print(FOLDER_LIST)

create_folders(CLEAN_FOLDER, FOLDER_LIST)
process_images(RAW_FOLDER, CLEAN_FOLDER, FOLDER_LIST)


# In[39]:


import os
import cv2
import numpy as np
from six.moves import cPickle as Pickle
import csv

DATA_FOLDER = "D:\\project\\try2"
image_size = 32
pixel_depth = 255
pickle_extension = '.pickle'
num_classes = 50
image_per_class = 200


def get_folders(path):
    data_folders = [os.path.join(path, d) for d in sorted(os.listdir(path))
                    if os.path.isdir(os.path.join(path, d))]

    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))

    return data_folders


def load_letter(folder, min_num_images):
    """Load the data for a single letter label."""
    image_files = os.listdir(folder)
    dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
    print(folder)
    image_index = -1
    for image_index, image in enumerate(image_files):
        image_file = os.path.join(folder, image)
        try:
            image_data = 1 * (cv2.imread(image_file, cv2.IMREAD_UNCHANGED).astype(float) > pixel_depth / 2)
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
        except IOError as err:
            print('Could not read:', image_file, ':', err, '- it\'s ok, skipping.')

    num_images = image_index + 1
    dataset = dataset[0:num_images, :, :]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (num_images, min_num_images))

    print('Full dataset tensor:', dataset.shape)
    print('Mean:', np.mean(dataset))
    print('Standard deviation:', np.std(dataset))
    return dataset


def maybe_pickle(data_folders, min_num_images_per_class, force=False):
    dataset_names = []
    for folder in data_folders:
        set_filename = folder + pickle_extension
        dataset_names.append(folder)
        if os.path.exists(set_filename) and not force:
            # You may override by setting force=True.
            print('%s already present - Skipping pickling.' % set_filename)
        else:
            # print('Pickling %s.' % set_filename)
            dataset = load_letter(folder, min_num_images_per_class)
            try:
                with open(set_filename, 'wb') as f:
                    Pickle.dump(dataset, f, Pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)

    return dataset_names


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def merge_datasets(pickle_files, train_size, test_size=0, valid_size=0):
    num_classes = len(pickle_files)
    print(num_classes)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    test_dataset, test_labels = make_arrays(test_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    valid_size_per_class = valid_size // num_classes
    test_size_per_class = test_size // num_classes
    train_size_per_class = train_size // num_classes

    print(valid_size_per_class, test_size_per_class, train_size_per_class)

    start_valid, start_test, start_train = 0, valid_size_per_class, (valid_size_per_class + test_size_per_class)
    end_valid = valid_size_per_class
    end_test = end_valid + test_size_per_class
    end_train = end_test + train_size_per_class

    print(start_valid, end_valid)
    print(start_test, end_test)
    print(start_train,end_train)

    s_valid, s_test, s_train = 0, 0, 0
    e_valid, e_test, e_train = valid_size_per_class, test_size_per_class, train_size_per_class
    temp = []
    for label, pickle_file in enumerate(pickle_files):
        temp.append([label, pickle_file[-4:]])
        try:
            with open(pickle_file + pickle_extension, 'rb') as f:
                letter_set = Pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:end_valid, :, :]
                    valid_dataset[s_valid:e_valid, :, :] = valid_letter
                    valid_labels[s_valid:e_valid] = label
                    s_valid += valid_size_per_class
                    e_valid += valid_size_per_class

                if test_dataset is not None:
                    test_letter = letter_set[start_test:end_test, :, :]
                    test_dataset[s_test:e_test, :, :] = test_letter
                    test_labels[s_test:e_test] = label
                    s_test += test_size_per_class
                    e_test += test_size_per_class

                train_letter = letter_set[start_train:end_train, :, :]
                train_dataset[s_train:e_train, :, :] = train_letter
                train_labels[s_train:e_train] = label
                s_train += train_size_per_class
                e_train += train_size_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    with open('classes.csv', 'w') as my_csv:
        writer = csv.writer(my_csv, delimiter=',')
        writer.writerows(temp)
    return valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels


data_folders = get_folders(DATA_FOLDER)
train_datasets = maybe_pickle(data_folders, image_per_class, True)
train_size = int(image_per_class * num_classes * 0.7)
test_size = int(image_per_class * num_classes * 0.2)
valid_size = int(image_per_class * num_classes * 0.1)

valid_dataset, valid_labels, test_dataset, test_labels, train_dataset, train_labels = merge_datasets(
    train_datasets, train_size, test_size, valid_size)

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)

pickle_file = 'data.pickle'

try:
    f = open(pickle_file, 'wb')
    save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
    }
    Pickle.dump(save, f, Pickle.HIGHEST_PROTOCOL)
    f.close()
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)


# In[2]:


import keras
import numpy as np
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from six.moves import cPickle as Pickle

batch_size = 128
num_classes = 50
epochs = 12

pickle_file = "data.pickle"

with open(pickle_file, 'rb') as f:
    save = Pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    # print('Test set', test_dataset.shape, test_labels.shape)


def reformat(dataset, labels):
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    labels = (np.arange(num_classes) == labels[:, None]).astype(np.float32)
    return dataset, labels


train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Testing set', test_dataset.shape, test_labels.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(32, 32, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_dataset, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(valid_dataset, valid_labels))
score = model.evaluate(test_dataset, test_labels, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




# In[3]:


import numpy as np
from numpy import genfromtxt
import csv
import operator
from keras.models import load_model
import cv2


# In[32]:


model = load_model("model.h5")

image = cv2.imread("D:\\project\\try\\has\\has.png", cv2.IMREAD_UNCHANGED)
"""if image.shape[2] == 4:
    image = read_transparent_png(image)"""
image = clean(image)
#cv2.imshow('gray', image)
#cv2.waitKey(0)

def predict(img):
    image_data = img
    dataset = np.asarray(image_data)
    dataset = dataset.reshape((-1, 32, 32, 1)).astype(np.float32)
    print(dataset.shape)
    a = model.predict(dataset)[0]

    classes = np.genfromtxt('classes.csv', delimiter=',')[:, 0].astype(int)

    print(classes)
    new = dict(zip(classes, a))
    res = sorted(new.items(), key=operator.itemgetter(1), reverse=True)

    print("#########***#########")
    print("Imagefile = ", image)
    print("Character = ", int(res[0][0]))
    print("Confidence = ", res[0][1] * 100, "%")
    if res[0][1] < 1:
        print("Other predictions")
        for newtemp in res:
            print("Character = ", newtemp[0])
            print("Confidence = ", newtemp[1] * 100, "%")


predict(image)


# In[49]:


print(words)


# In[50]:


print(words[14])


# In[ ]:




