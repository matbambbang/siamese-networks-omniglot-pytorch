import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import argparse
import os
from model import *
from dataloader import *

#torch.cuda.set_device(1)

def train(model,train_data,valid_data,EPOCH,path) :
    #optimizer = optim.Adam([
    #    {'params': model.layer1.parameters(), 'weight_decay': 0.5},
    #    {'params': model.layer2.parameters(), 'weight_decay': 0.5},
    #    {'params': model.layer3.parameters(), 'weight_decay': 0.5},
    #    {'params': model.layer4.parameters(), 'weight_decay': 0.5},
    #    {'params': model.linear.parameters(), 'weight_decay': 0.5},
    #    {'params': model.final.parameters(), 'weight_decay': 0.5}
    #    ], lr=0.0001)
    #optimizer = optim.Adam(model.parameters(), lr=0.0005)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
    loss_function = nn.BCELoss()
    train_loss_list = []
    validation_loss_list = []
    validation_acc_list = []
    train_batch_size = 128
    valid_batch_size = 20

    for epoch in range(EPOCH) :
        print("Epoch %d" %(epoch+1))
        # set learning rate & momentum
        scheduler.step()

        loss_arr = []
        model.train()
        model.batch_size = train_batch_size
        
        # train
        for i, (images,labels) in enumerate(train_data) :
            model.batch_size = train_batch_size

            image1 = []
            image2 = []
            for j in range(images.size(0)) :
                image1.append(images[j][0].view(1,1,images.size(2),images.size(3)))
                image2.append(images[j][1].view(1,1,images.size(2),images.size(3)))
            image1 = Variable(torch.cat(image1)).cuda()
            image2 = Variable(torch.cat(image2)).cuda()
            labels = Variable(labels.type(torch.FloatTensor),requires_grad=False).view(-1,1).cuda()

            scores = model(image1,image2)
            loss = loss_function(scores,labels)
            loss_arr.append(loss.cpu().data[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #if i%50 == 0 :
            #    print("Iter %d Train loss : %.5f" %(i,loss.cpu().data[0]))

        print("Train mean loss(train) : %.5f" %(np.array(loss_arr).mean()))
        train_loss_list.append(np.array(loss_arr).mean())

        # validation
        num_sample = 0
        num_correct = 0
        loss_arr = []
        model.eval()
        model.batch_size = valid_batch_size

        for i, (images,labels) in enumerate(valid_data) :
            model.batch_size = valid_batch_size
            model.eval()

            image1 = []
            image2 = []
            for j in range(images.size(0)) :
                image1.append(images[j][0].view(1,1,images.size(2),images.size(3)))
                image2.append(images[j][1].view(1,1,images.size(2),images.size(3)))
            image1 = Variable(torch.cat(image1)).cuda()
            image2 = Variable(torch.cat(image2)).cuda()
            labels = Variable(labels.type(torch.FloatTensor)).view(-1,1).cuda()
            #targets = Variable(labels.type(torch.FloatTensor)).view(-1,1).cuda()

            scores = model(image1,image2)
            loss = loss_function(scores,labels)
            #loss = loss_function(targets,scores)
            loss_arr.append(loss.cpu().data[0])
            
            num_sample += 1
            inference = scores.cpu().data.numpy()
            answer = labels.cpu().data.numpy()
            if np.argmax(inference) == np.argmax(answer) :
                num_correct += 1
        print("Validation loss : %.5f" %(np.array(loss_arr).mean()))
        print("Validation accuracy : %.3f" %(float(num_correct/num_sample)))
        validation_loss_list.append(np.array(loss_arr).mean())
        validation_acc_list.append(float(num_correct/num_sample))
        #if min(validation_loss_list) >= np.array(loss_arr).mean() :
        param_path = os.path.join(path,'epoch'+str(epoch)+'param.param')
        torch.save(model.state_dict(),param_path)

        print("")

    print("Learning Finished")
    print("Final Train Loss : %.5f" %(train_loss_list[len(train_loss_list)-1]))
    print("Final Validation Loss : %.5f" %(validation_loss_list[len(validation_loss_list)-1]))
    print("Final Validation Accuracy : %.3f\n" %(validation_acc_list[len(validation_acc_list)-1]))

    return model, train_loss_list, validation_loss_list, validation_acc_list

def test(model,test_data) :
    model.eval()
    model.batch_size = 20

    num_sample = 0
    num_correct = 0

    for i, (images,labels) in enumerate(test_data) :
        model.batch_size = 20
        model.eval()

        image1 = []
        image2 = []
        for i in range(images.size(0)) :
            image1.append(images[i][0].view(1,1,images.size(2),images.size(3)))
            image2.append(images[i][1].view(1,1,images.size(2),images.size(3)))
        image1 = Variable(torch.cat(image1)).cuda()
        image2 = Variable(torch.cat(image2)).cuda()

        labels = Variable(labels.type(torch.FloatTensor)).view(-1,1).cuda()
        scores = model(image1,image2)

        num_sample += 1
        inference = scores.cpu().data.numpy()
        answer = labels.cpu().data.numpy()
        if np.argmax(inference) == np.argmax(answer) :
            num_correct += 1

    test_acc = float(num_correct/num_sample)
    print("Test Accuracy : %.3f\n" %(test_acc))

    return test_acc

def main(image_path,batch_size,EPOCH,seed) :
    torch.manual_seed(seed)
    model = Model(batch_size)
    model.cuda()
    # Call train_data, test_data
    with open(os.path.join(image_path,'train_loader_transform.pkl'),'rb') as f :
        train_data = pickle.load(f)
    with open(os.path.join(image_path,'valid_loader_transform.pkl'),'rb') as f :
        valid_data = pickle.load(f)
    with open(os.path.join(image_path,'test_loader_transform.pkl'),'rb') as f :
        test_data = pickle.load(f)

    model, train_loss, validation_loss, validation_acc = train(model,train_data,valid_data,EPOCH,image_path)
    test_acc = test(model,test_data)



if  __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./data',
                        help='Image path')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--EPOCH', type=int, default=200,
                        help='max epoch')
    parser.add_argument('--seed', type=int, default=0,
                        help='torch seed')
    args = parser.parse_args()

    image_path = args.image_path
    batch_size = args.batch_size
    EPOCH = args.EPOCH
    seed = args.seed
    main(image_path,batch_size,EPOCH,seed)
