import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

from algorithm.model import CNN
from algorithm.loss import loss_drop_gaussian
from common.utils import accuracy


from .circle_loss import convert_label_to_similarity, CircleLoss
import math

from .contrastive_loss import ContrastiveLoss


class Sl_loss:
    def __init__(self, args, train_dataset, device, input_channel, num_class):

        # Hyper Parameters
        self.batch_size = args.batch_size
        learning_rate = args.lr
        self.lr = learning_rate
        args.noise_rate_ = args.noise_rate

        if args.noise == 'ood':
            args.noise_rate_ = train_dataset.actual_noise_rate

        if args.forget_rate is None:
            if args.noise_type == "asymmetric":
                forget_rate = args.noise_rate_ / 2
            else:
                forget_rate = args.noise_rate_
        else:
            forget_rate = args.forget_rate
        

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule forget_rate小到大 remember rate大到小
        
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual) 
        
      


        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset

        self.model = CNN(input_channel=input_channel, n_outputs=num_class).to(device)

        if learning_rate == 0.0001:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=1e-3,momentum=0.9)


        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[40, 80], gamma=0.1)
        

        self.loss_fn = loss_drop_gaussian   



        self.adjust_lr = args.adjust_lr
        self.num_gradual = args.num_gradual

    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model.eval()  # Change model to 'eval' mode.

        correct = 0
        total = 0
        for _, (images, labels, _) in enumerate(test_loader):
            

            images = Variable(images).to(self.device)
            logits, _ = self.model(images)  


            outputs = F.softmax(logits, dim=1)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred.cpu() == labels).sum()


        acc = 100 * float(correct) / float(total)
        return acc

        # Train the Model

    def train(self, train_loader, epoch):
        print('Training ...')
        self.model.train()  # Change model to 'train' mode.

        if self.lr == 0.0001:
            self.adjust_learning_rate(self.optimizer, epoch)

        # self.learning_schedule.step()
        train_total = 0
        train_correct = 0
        pure_ratio_1_list = []

        for i, (images, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            if i > self.num_iter_per_epoch:
                break
            
            if len(images[0].shape) == 3:
                weak_images = images[0].unsqueeze(0)  
                strong_images = images[1].unsqueeze(0)  
            else:
                weak_images = images[0]  
                strong_images = images[1] 


            weak_images = Variable(weak_images).to(self.device)
            strong_images = Variable(strong_images).to(self.device)
            labels = Variable(labels).to(self.device) 



            # Forward + Backward + Optimize
            logits1, normalized_feature= self.model(weak_images)
            logits2, normalized_feature2= self.model(strong_images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1


            if epoch<self.num_gradual: 
                loss_1, pure_ratio_1, ind_update, ind_noisy = self.loss_fn(logits1, labels, self.rate_schedule[epoch],
                                                ind, self.noise_or_not, after_warmup=False)  # self.rate_schedule[epoch]表示 forget_rate
            
            
     
                
            
            
            else: 
                loss_1, pure_ratio_1, ind_update, ind_noisy = self.loss_fn(logits1, labels, self.rate_schedule[epoch],
                                                ind, self.noise_or_not, after_warmup=True)  # self.rate_schedule[epoch]表示 forget_rate

                        
               

 
            # circle_loss
            if epoch>=self.num_gradual:
                criterion = CircleLoss(m=0.5, gamma=10)
                circle_loss = criterion(*convert_label_to_similarity(normalized_feature[ind_update], labels[ind_update]))
                if circle_loss==0:
                    loss_1 = loss_1
                else:
                    loss_1 = loss_1  + math.log(circle_loss) 


                contrastive_loss=ContrastiveLoss(batch_size=normalized_feature[ind_noisy].size(0))
                contrastive_loss = contrastive_loss(normalized_feature[ind_noisy], normalized_feature2[ind_noisy])
                loss_1 = loss_1  + contrastive_loss 


            self.optimizer.zero_grad()
            loss_1.backward()
            self.optimizer.step()

            pure_ratio_1_list.append(100 * pure_ratio_1)

            if (i + 1) % self.print_freq == 0:
                print(
                    'Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f,  Pure Ratio %.4f %% '
                    % (epoch + 1, self.n_epoch, i + 1, len(self.train_dataset) // self.batch_size, prec1,
                       loss_1.data.item(), sum(pure_ratio_1_list) / len(pure_ratio_1_list)))

        train_acc1 = float(train_correct) / float(train_total)

        if self.lr != 0.0001:
            self.scheduler.step()

        return train_acc1, pure_ratio_1_list

    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]  
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  
