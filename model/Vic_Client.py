import copy

from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.Network import NN, MLP_SketchLinear, CNNCifar, CNNMnist, CNNCifar_Sketch, CNNMnist_Sketch

from src.conf import Args

args = Args()
class DatasetSplit(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        image, label = self.dataset[self.idx[item]]
        return image, label


class VicClient:
    def __init__(self, data, idx, args):
        self.idx = idx
        self.args = args
        self.loss_func = torch.nn.NLLLoss()
        self.ldr_train = DataLoader(DatasetSplit(data, idx), batch_size=self.args.local_batch_size, shuffle=True)

        if self.args.model_type == 'NN':
            self.model = NN(self.args.dim_in, self.args.dim_out).to(self.args.device)
        elif self.args.model_type == 'MLP_SketchLinear':
            self.model = MLP_SketchLinear(self.args.dim_in, self.args.dim_out, self.args.p).to(self.args.device)

        # model == CNN
        elif self.args.model_type == 'CNN' and self.args.datatype == 'mnist':
            self.model = CNNMnist().to(self.args.device)
        elif self.args.model_type == 'CNN' and self.args.datatype == 'cifar':
            self.model = CNNCifar().to(self.args.device)

        elif self.args.model_type == 'CNN' and self.args.datatype == 'LFW':
            self.model = CNNCifar(num_classes = 2).to(self.args.device)

        # model == CNN_Sketch
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'mnist':
            self.model = CNNMnist_Sketch(self.args.p).to(self.args.device)
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'cifar':
            self.model = CNNCifar_Sketch(self.args.p).to(self.args.device)
        
        elif self.args.model_type == 'CNN_sketch' and self.args.datatype == 'LFW':
            self.model = CNNCifar_Sketch(num_classes = 2).to(self.args.device)

    # get gradients and sketch matrix S from server
    # for gaussian sketch, one needs to pass in sketch_matrices
    # for count sketch, one only needs to pass in hash indices and rnadom signs of the count sketch matrix
    
    def get_paras(self, paras, hash_idxs, rand_sgns, sketch_matrices=None):
        if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
            if self.args.sketchtype == "gaussian":
                self.sketch_matrices = sketch_matrices
            else:
                self.hash_idxs = hash_idxs
                self.rand_sgns = rand_sgns
            self.prev_paras = paras
            self.model.load_state_dict(paras)
        else:
            self.prev_paras = paras
            self.model.load_state_dict(paras)
    
    # get average updates from server avg_up = (up1 + up2)/2
    def get_avg_updates(self, avg_update):
            self.broadcasted_update = avg_update  # avg_updates from server

    def adjust_learning_rate(self, optimizer, epoch, step):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr_ad = self.args.learningrate_client * (0.1 ** (epoch // step))
        if lr_ad <= 1e-5:
            lr = 1e-5
        else:
            lr = lr_ad
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # every client sends gradients to server after training on local data for several epochs
    # rather than send the gradients, it actually send the updates
    def send_grads(self):
        current_grad = dict()
        current_paras = self.model.state_dict()
        for k in current_paras.keys():
            current_grad[k] = current_paras[k] - self.prev_paras[k]
        return current_grad

    def send_paras(self):
        return copy.deepcopy(self.model.state_dict())

    def size(self):
        return len(self.idx)

    # local training for each client
    # optimizer: Adam
    def train(self, current_round):
        self.model.train()
        # train and update

        epoch_losses = list()
        epoch_acces = list()

        if args.attack == 1:
            optimizer = torch.optim.SGD(self.model.parameters(), momentum=0, lr=self.args.learningrate_client)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learningrate_client)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)
        # scheduler = ExponentialLR(optimizer, 0.9, last_epoch=-1)

        print("")
        print("======================")
        for iter in range(self.args.local_epochs):
            print('Victim client local epoch', iter)
            l_sum = 0.0
            correct = 0.0
            # batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if args.attack==1 and batch_idx >0: 
                    break

                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                if self.args.model_type == 'MLP_SketchLinear' or self.args.model_type == 'CNN_sketch':
                    if self.args.sketchtype == 'gaussian':
                        log_probs = self.model(images, sketchmats=self.sketch_matrices)
                    else:
                        log_probs = self.model(images, self.hash_idxs, self.rand_sgns)
                else:
                    # CNN model
                    log_probs = self.model(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                l_sum += loss.item()
                pred = log_probs.data.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum()
                
                # only save the first image
                if current_round == 0:
                    # save original_image
                    plt.figure(figsize=(4, 4))   
                    original_image = images[0].cpu().permute(1, 2, 0).detach().numpy()
                    plt.imshow(original_image)
                    plt.title('Original_image')
                    save_path = './data/adv_attack_res/Original_img'
                    print("Successfully saved original image in ", save_path)
                    plt.savefig(save_path)

            n = float(len(self.idx))
            epoch_acc = 100.0 * float(correct) / n
            epoch_loss = l_sum / (batch_idx + 1)
            epoch_losses.append(epoch_loss)
            epoch_acces.append(epoch_acc)
        return sum(epoch_losses) / len(epoch_losses), sum(epoch_acces) / len(epoch_acces)


