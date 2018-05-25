import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import argparse
import os
import random
import pickle
from imgclass import Images
from imgclass import SummaryInfo


"""
image = Images(dir)
    image.language  : language, str
    image.char      : character, str
    image.drawer    : drawer, int
    image.get_info  : store image information, image.get_info(language,char,drawer)
    image.get_image : get image tensor (1*105*105), image.get_image(transform)

info = SummaryInfo(dir)
    info.background : background language list
    info.evaluation : evaluation language list
    info.char       : language's character list, dict

"""

def drawer_separation() :
    drawer = [i for i in range(1,21)]
    random.shuffle(drawer)

    drawer_train = drawer[:12]
    drawer_valid = drawer[12:16]
    drawer_test = drawer[16:]

    return drawer_train, drawer_valid, drawer_test

def eval_separation() :
    info = SummaryInfo()

    eval_list = info.evaluation
    random.shuffle(eval_list)

    valid_lgg = eval_list[:10]
    test_lgg = eval_list[10:]

    return valid_lgg, test_lgg


def preprocess_train(drawer_train, dir_name='./Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', num=30000) :
    info = SummaryInfo()

    # 30,000 pairs - 15,000 true, 15,000 false
    pairs = []

    for i in range(int(num/2)) :
        language = random.choice(info.background)
        char = random.choice(info.char[language])
        drawer1, drawer2 = np.random.choice(drawer_train, 2, replace=False)

        pairs.append([language, language, char, char, drawer1, drawer2])

    for i in range(int(num/6)) :
        language = random.choice(info.background)
        char1 = random.choice(info.char[language])
        char2 = random.choice(info.char[language])
        while char2 == char1 :
            char2 = random.choice(info.char[language])
        assert char1 != char2
        drawer1, drawer2 = np.random.choice(drawer_train, 2)

        pairs.append([language, language, char1, char2, drawer1, drawer2])

    for i in range(int(num/3)) :
        language1 = random.choice(info.background)
        language2 = random.choice(info.background)
        while language2 == language1 :
            language2 = random.choice(info.background)
        char1 = random.choice(info.char[language1])
        char2 = random.choice(info.char[language2])
        drawer1, drawer2 = np.random.choice(drawer_train, 2)

        pairs.append([language1, language2, char1, char2, drawer1, drawer2])

    random.shuffle(pairs)
    images = []
    labels = []

    for elem in pairs :
        # elem = [language1, language2, char1, char2, drawer1, drawer2]
        if elem[0] == elem[1] and elem[2] == elem[3] :
            labels.append(torch.FloatTensor([1.]))
        else :
            labels.append(torch.FloatTensor([0.]))
        img1 = Images('background')
        img2 = Images('background')
        img1.get_info(elem[0], elem[2], elem[4])
        img2.get_info(elem[1], elem[3], elem[5])

        # You can define your own transformations on this part
        temp1 = img1.get_image()
        temp2 = img2.get_image()
        # custom transformation
        #transform = transforms.Compose([
        #    transforms.RandomResizedCrop(105, scale=(0.7,1.3)),
        #    transforms.Grayscale(),
        #    transforms.ToTensor()
        #])
        #temp1 = img1.get_image(transform)
        #temp2 = img2.get_image(transform)
        assert temp1.shape == temp2.shape

        shapes = [1]
        shapes.extend(list(temp1.shape))
        images.append(torch.cat([temp1.view(shapes),temp2.view(shapes)],1))

    train_images = torch.cat(images)
    train_labels = torch.cat(labels)

    return train_images, train_labels, pairs

def preprocess_verification(drawer_evaltype, lgg_list, dir_name='./Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', num=400) :
    info = SummaryInfo()

    # 400 pairs
    pairs = []

    for i in range(int(num/2)) :
        language = random.choice(lgg_list)
        char = random.choice(info.char[language])
        drawer1, drawer2 = np.random.choice(drawer_evaltype, 2, replace=False)

        pairs.append([language, language, char, char, drawer1, drawer2])

    for i in range(int(num/2)) :
        language1 = random.choice(info.evaluation)
        language2 = random.choice(info.evaluation)
        char1 = random.choice(info.char[language1])
        char2 = random.choice(info.char[language2])
        while language1 == language2 and char1 == char2 :
            language2 = random.choice(info.evaluation)
            char2 = random.choice(info.char[language2])
        
        drawer1, drawer2 = np.random.choice(drawer_evaltype, 2)

        pairs.append([language1, language2, char1, char2, drawer1, drawer2])

    random.shuffle(pairs)
    images = []
    labels = []

    for elem in pairs :
        # elem = [language1, language2, char1, char2, drawer1, drawer2]
        if elem[0] == elem[1] and elem[2] == elem[3] :
            labels.append(torch.FloatTensor([1.]))
        else :
            labels.append(torch.FloatTensor([0.]))
        img1 = Images('evaluation')
        img2 = Images('evaluation')
        img1.get_info(elem[0], elem[2], elem[4])
        img2.get_info(elem[1], elem[3], elem[5])

        temp1 = img1.get_image()
        temp2 = img2.get_image()
        assert temp1.shape == temp2.shape

        shapes = [1]
        shapes.extend(list(temp1.shape))
        images.append(torch.cat([temp1.view(shapes),temp2.view(shapes)],1))

    verification_images = torch.cat(images)
    verification_labels = torch.cat(labels)

    return verification_images, verification_labels, pairs

def preprocess_eval(drawer_evaltype, lgg_list, dir_name='./Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', num=400) :
    info = SummaryInfo()

    # 400 pairs
    pairs = []

    for i in range(num) :
        language = random.choice(lgg_list)
        char_list = np.random.choice(info.char[language],20,replace=False)
        char_true = np.random.choice(char_list)
        drawer1, drawer2 = np.random.choice(drawer_evaltype, 2, replace=False)
        pair = [[language, language, char_true, char, drawer1, drawer2] for char in char_list]
        pairs.append(pair)
    
    random.shuffle(pairs)
    images = []
    labels = []

    for sets in pairs :
        for elem in sets :
            label = []
            image = []
            if elem[0] == elem[1] and elem[2] == elem[3] :
                label.append(torch.FloatTensor([1.]))
            else :
                label.append(torch.FloatTensor([0.]))
            img1 = Images('evaluation')
            img2 = Images('evaluation')
            img1.get_info(elem[0], elem[2], elem[4])
            img2.get_info(elem[1], elem[3], elem[5])

            # You can define your own trasnformations on this part
            temp1 = img1.get_image()
            temp2 = img2.get_image()

            shapes = [1]
            shapes.extend(list(temp1.shape))
            image.append(torch.cat([temp1.view(shapes), temp2.view(shapes)],1))

            labels.append(torch.cat(label))
            images.append(torch.cat(image))

    eval_labels = torch.cat(labels)
    eval_images = torch.cat(images)

    return eval_images, eval_labels, pairs

def check_preprocess(dir_name = './Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', train_num=30000, eval_num=400) :
    drawer_train, drawer_valid, drawer_test = drawer_separation()
    valid_list, test_list = eval_separation()

    train_images, train_labels, train_pairs = preprocess_train(drawer_train, dir_name, train_num)
    for i, elem in enumerate(train_pairs) :
        if elem[0] == elem[1] and elem[2] == elem[3] :
            assert elem[4] != elem[5], "Preprocess Error - Same images sampled."
            assert int(train_labels[i]) != 0, "Preprocess Error - Wrong labeled sample observed(true 1, labeled 0)."
        else :
            assert int(train_labels[i]) == 0, "Preprocess Error - Wrong labeled sample observed(true 0, labeled 1)."

    verification_images, verification_labels, verification_pairs = preprocess_verification(drawer_valid, valid_list, dir_name, eval_num)

    valid_images, valid_labels, valid_pairs = preprocess_eval(drawer_valid, valid_list, dir_name, eval_num)
    assert valid_images.size(0) == 8000, "Preprocess Error - Wrong counts."

    test_images, test_labels, test_pairs = preprocess_eval(drawer_test, test_list, dir_name, eval_num)
    assert test_images.size(0) == 8000, "Preprocess Error - Wrong counts."

    return train_images, train_labels, verification_images, verification_labels, valid_images, valid_labels, test_images, test_labels



def preprocess(dir_name='./Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', train_num=30000, eval_num=400) :
    train_images, train_labels, verification_images, verification_labels, valid_images, valid_labels, test_images, test_labels = check_preprocess(dir_name, train_num, eval_num)

    train_dataset = TensorDataset(train_images, train_labels)
    verification_dataset = TensorDataset(verification_images, verification_labels)
    valid_dataset = TensorDataset(valid_images, valid_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, drop_last=True)
    verification_loader = DataLoader(verification_dataset, batch_size=20, shuffle=False, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    return train_loader, verification_loader, valid_loader, test_loader

def save_loader(save_name=None, dir_name='./Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset', train_num=30000, eval_num=400) :
    train_loader, verification_loader, valid_loader, test_loader = preprocess(dir_name, train_num, eval_num)
    if save_name :
        with open('./data/train_loader_'+save_name+'.pkl','wb') as f :
            pickle.dump(train_loader,f)
        with open('./data/valid_loader_'+save_name+'.pkl','wb') as f :
            pickle.dump(valid_loader,f)
        with open('./data/test_loader_'+save_name+'.pkl','wb') as f :
            pickle.dump(test_loader,f)
    else :
        with open('./data/train_loader.pkl','wb') as f :
            pickle.dump(train_loader,f)
        with open('./data/verification_loader.pkl','wb') as f :
            pickle.dump(verification_loader,f)
        with open('./data/valid_loader.pkl','wb') as f :
            pickle.dump(valid_loader,f)
        with open('./data/test_loader.pkl','wb') as f :
            pickle.dump(test_loader,f)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,
                        default = './Siamese-Networks-for-One-Shot-Learning/Omniglot Dataset',
                        help = 'directory of image files')

    args = parser.parse_args()

    save_loader(dir_name = args.dir)
