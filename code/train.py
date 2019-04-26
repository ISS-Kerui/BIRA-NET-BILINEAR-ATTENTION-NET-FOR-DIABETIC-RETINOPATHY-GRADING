# encoding: utf-8
#author: Zhang Kerui
#date: Jun 02 2019
import os
import torch
import random
import argparse
from model import KeNet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datagenerator import ImageFolder
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
# Training settings
class InterclassLoss(nn.Module):
    def __init__(self):
        super(InterclassLoss, self).__init__()
    def forward(self, pred, truth):
        loss = 0.0
        # loss = Variable(loss.data, requires_grad=True)
        pred = pred.type(torch.FloatTensor)
        _, predicted = torch.max(pred.data, 1)
        
        predicted = torch.Tensor(predicted.type(torch.FloatTensor))
        predicted = predicted.cuda()
        pred = pred.cuda()
        
        for i, p in enumerate(pred):
            for j in range(0,5):
                if j == truth[i]:
                    M = self._getM(truth[i]).cuda()
            
                    weight = (abs(predicted[i]-truth[i])+1)/(M.cuda())
                    loss += weight*(-p[j])
        loss = loss/len(pred)
        loss = Variable(loss, requires_grad=True)
        return loss

    
    def _getM(self,label):
        M = 0.0
        for i in range(0,5):
            M += abs(label-i)+1
        return M

        
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, default = 30)
parser.add_argument('--epochs', type = int, default = 10)
parser.add_argument('--lr', type = float, default = 0.0002)
parser.add_argument('--no-cuda', action = 'store_true', default = False)
parser.add_argument('--seed', type = int, default = 1)
parser.add_argument('--save-epoch', type=int, default=5)
parser.add_argument('--weight-decay', type=float, default=1e-8)
parser.add_argument('--image-size', type = int, default = 610)
checkpoint_dir = 'output/weights'
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.image_size = 610
class_num = 5
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = KeNet(class_num)
if args.cuda:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


data_dir = '/home/kerui/data/new_eye/data/train'
valid_dir = '/home/kerui/data/new_eye/data/valid'
writer = SummaryWriter()

def train(path):
    model.train()
    if path:
        model.load_state_dict(torch.load(path))
    unchanged_params = list(map(id, model.features[:-2].parameters()))
    unchanged_params += list(map(id, model.up_c2.parameters()))
    training_params = filter(lambda p: id(p) not in unchanged_params, model.parameters())
    for param in model.up_c2.parameters():
        param.requires_grad = False
    for param in model.features[:-2].parameters():
        param.requires_grad = False
    optimizer = optim.SGD(training_params, lr=args.lr,momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=6, verbose=True)

    trainset, dataiter = ImageFolder(data_dir, args.image_size, args.batch_size, 'train')
    epoch_size = len(trainset) // args.batch_size

    print('开始训练')
    best_accuracy = 0.
    best_epoch = 0
    end_patient = 0

    for e in range(args.epochs):
        batch_iterator = iter(dataiter)
        progress_bar = tqdm(range(epoch_size))
        loss_sum = 0.
        correct_num = 0
        cnt = 0
        for i in progress_bar:
            images, labels = next(batch_iterator)
            images, labels = Variable(images, requires_grad=True), Variable(labels)

            images, labels = images.to(device), labels.to(device)
            cnt += labels.size(0)

            logits, softmax = model(images)

            _, predicted = torch.max(logits, 1)
            batch_correct = (predicted == labels).sum()
            correct_num += batch_correct

            optimizer.zero_grad()

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, labels)
            # loss = loss_fn(softmax, labels)
            loss_sum += loss.data.cpu().numpy()

            loss.backward()

            optimizer.step()
        accuracy = correct_num.item() / cnt
        print("epoch[%d],loss: %.4f, accuracy:%.4f." % (e + 1, loss_sum / epoch_size, accuracy))
        writer.add_scalar('data_2/loss', loss_sum/epoch_size, e+1)
        writer.add_scalar('data_2/accuracy', accuracy, e+1)
        
        # 输出test的准确率
        test_accuracy = test(test_data_dir)
        print('the accuracy of test is: %.4f' % test_accuracy)
        writer.add_scalar('data_2/test_accuracy', test_accuracy, e+1)


        scheduler.step(test_accuracy)

        if test_accuracy > best_accuracy:
            model_file = os.path.join(checkpoint_dir, 'train_all_epoch_%03d_acc_%.4f.pth' %
                                      (best_epoch, best_accuracy))
            if os.path.isfile(model_file):
                os.remove(model_file)

            end_patient = 0
            best_accuracy = test_accuracy
            best_epoch = e + 1
            print('保存权值')
            torch.save(model.cpu().state_dict(), os.path.join(checkpoint_dir, 'train_all_epoch_%03d_acc_%.4f.pth' %
                                                            (best_epoch, best_accuracy)))
            

            model.to(device)
        
        else:
            end_patient += 1

        if end_patient >= 10:
            break

      

def test(test_data_dir):
    with torch.no_grad():
        model.eval()
        # model.load_state_dict(torch.load(path))

        testset, dataiter = ImageFolder(test_data_dir, args.image_size, args.batch_size, 'test')
        correct_num = 0
        cnt = 0
        for test_images, test_labels in dataiter:

            test_images, test_labels = test_images.to(device), test_labels.to(device)

            cnt += test_labels.size(0)

            logits, _ = model(test_images)

            _, predicted = torch.max(logits, 1)

            correct = (predicted == test_labels).sum()
            correct_num += correct

        accuracy = correct_num.item() / cnt
    print(correct_num.item())
    print(cnt)
    model.train()
    return accuracy
if __name__ == '__main__':
    train(None)
# accuracy,output,target = eval('./output/after_60epoch/epoch060_iter100.pth')
# print output
# print target.shape
# num_label0 = np.sum(target==0)
# num_label1 = np.sum(target==1)
# num_label2 = np.sum(target==2)
# num_label3 = np.sum(target==3)
# num_label4 = np.sum(target==4)
# print num_label0,num_label1,num_label2,num_label3,num_label4
# correct_0 = 0
# correct_1 = 0
# correct_2 = 0
# correct_3 = 0
# correct_4 = 0
# for i in range(len(output)):
#     if output[i] == target[i] ==0:
#         correct_0 +=1
#     elif output[i] == target[i] ==1:
#         correct_1 +=1
#     elif output[i] == target[i] ==2:
#         correct_2 +=1
#     elif output[i] == target[i] ==3:
#         correct_3 +=1
#     elif output[i] == target[i] ==4:
#         correct_4 +=1
# print correct_0/(num_label0+0.0)
# print correct_1/(num_label1+0.0)
# print correct_2/(num_label2+0.0)
# print correct_3/(num_label3+0.0)
# print correct_4/(num_label4+0.0)