import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
#from readp import *

# Define data values
str1 = "accs_CNN_sketchcifar_lr_0.1target_acc_97_sketch_type_gaussian_p_2__hessa"
filename = "data/results/" + str1
acc = pickle.load(open(filename, 'rb'))
acc = torch.stack(acc)

#str1 = "accs_cifarCNNcifar_lr_0.1target_acc_97"
#filename = "data/results/" + str1
#acc1 = pickle.load(open(filename, 'rb'))
#acc1 = torch.stack(acc1)

#print([acci.cpu().item() for acci in acc])
str2 = "losses_CNN_sketchcifar_lr_0.1target_acc_97"
str2 = "losses_CNN_sketchcifar_lr_0.1target_acc_97_sketch_type_gaussian_p_2__hessa"
filename = "data/results/" + str2
loss = pickle.load(open(filename, 'rb'))
#loss = torch.stack(loss)

#str2 = "losses_cifarCNNcifar_lr_0.1target_acc_97"
#filename = "data/results/" + str2
#loss1 = pickle.load(open(filename, 'rb'))
#loss = torch.stack(loss)
#print(loss)

y = acc.cpu()
#y1 = acc1.cpu()
z = loss
#z1 = loss1

x = np.array(range(len(y)))+1
#x1 = np.array(range(len(y1)))+1


# Plot a simple line chart
fig = plt.figure(1)
ax1 = fig.add_subplot(211)
ax1.set_ylabel('Accuracy')
ax1.plot(x, y)
#ax1.plot(x1,y1)


ax2 = fig.add_subplot(212)
ax2.set_ylabel('Loss')
ax2.plot(x, z)
#ax2.plot(x1,z1)
ax2.set_xlabel('Communication rounds')


# Plot another line on the same chart/graph

plt.savefig("cifar_sketch_gausssian_q_2_hessa.png")