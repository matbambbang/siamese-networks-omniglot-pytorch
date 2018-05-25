import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
from PIL import Image
import random
import os

"""
Documentations(?)
data_type   : 'background' or 'evaluation'
transform   : torchvision.transforms
language    : language type, string
char        : character type, string
drawer      : drawer number, int

"""



class Images :
    def __init__(self, data_type, dir_path = './Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset') :
        self.data_type = data_type
        self.dir_path = dir_path

    def get_info(self, language, char, drawer) :
        self.language = language
        self.char = char
        if drawer < 10 :
            self.drawer = '0' + str(drawer)
        else :
            self.drawer = str(drawer)

    def get_image(self, transform=None) :
        dirname = 'images_' + self.data_type
        image_path = os.path.join(self.dir_path, dirname, self.language, self.char)
        image_list = os.listdir(image_path)
        filename = ''

        for image_file in image_list :
            if image_file[len(image_file)-6:len(image_file)-4] == self.drawer :
                filename = image_file
                break
        
        image = Image.open(os.path.join(image_path, filename))

        if transform :
            return transform(image)
        else :
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()
                ])
            return transform(image)

    def finder(self, key_type, info) :
        if key_type == 'language' and info == self.language :
            return True
        elif key_type == 'character' and info == self.char :
            return True
        elif key_type == 'drawer' and info == self.drawer :
            return True
        else :
            return False


class SummaryInfo :
    def __init__(self,dir_path = './Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset') :
        self.language_info(dir_path)
        self.char_info(dir_path)
        self.drawer = [i for i in range(1,21)]

    def language_info(self, dir_path) :
        # background
        self.background = os.listdir(os.path.join(dir_path, 'images_background'))
        self.evaluation = os.listdir(os.path.join(dir_path, 'images_evaluation'))

    def char_info(self, dir_path) :
        char_gnd = {language : os.listdir(os.path.join(dir_path, 'images_background', language)) for language in self.background}
        char_eval = {language : os.listdir(os.path.join(dir_path, 'images_evaluation', language)) for language in self.evaluation}

        self.char = char_eval.copy()
        self.char.update(char_gnd)
