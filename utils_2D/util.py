import os
import time
from sys import stderr
import pandas as pd
import torch


# Enter your wandb API key here
WANDB_API_KEY = "YOUR_WANDB_API_KEY_HERE"
#--------------------------------------- VTS ---------------------------------------
splits_VTS              = ['fold 0.txt', 'fold 1.txt', 'fold 2.txt', 'fold 3.txt', 'fold 4.txt']
gestures_VTS            = [f'G{i}' for i in range(0, 5+1)]
#------------------------------------- JIGSAWS -------------------------------------
splits_JIGSAWS          = ['data_B.csv', 'data_C.csv', 'data_D.csv', 'data_E.csv', 'data_F.csv', 'data_G.csv', 'data_H.csv', 'data_I.csv']
gestures_JIGSAWS        = ['G1', 'G5', 'G8', 'G2', 'G3', 'G6', 'G4', 'G11', 'G9', 'G10']
#---------------------------------- MultiBypass140 ----------------------------------
splits_MultiBypass140   = {'train':   ['data_train_0.csv', 'data_train_1.csv', 'data_train_2.csv', 'data_train_3.csv', 'data_train_4.csv'],
                           'val':     ['data_val_0.csv', 'data_val_1.csv', 'data_val_2.csv', 'data_val_3.csv', 'data_val_4.csv'],
                           'test':    ['data_test_0.csv', 'data_test_1.csv', 'data_test_2.csv', 'data_test_3.csv', 'data_test_4.csv']}
steps_MultiBypass140    = [f'S{i}' for i in range(0, 45+1)]
phases_MultiBypass140   = [f'P{i}' for i in range(0, 11+1)]
#-------------------------------------- RARP50 --------------------------------------
splits_SAR_RARP50       = ['data_0.csv', 'data_1.csv', 'data_2.csv', 'data_3.csv', 'data_4.csv']
gestures_SAR_RARP50     = [f'G{i}' for i in range(0, 7+1)]
#------------------------------------------------------------------------------------


def log(file, msg):
    """Log a message.

    :param file: File object to which the message will be written.
    :param msg:  Message to log (str).
    """
    if type(msg) == type(pd.DataFrame([])):
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: \n"), msg, file=stderr)
        file.write(time.strftime("[%d.%m.%Y %H:%M:%S]:\n"))
        file.write(msg.to_string())
        file.write(os.linesep)


    else:
        print(time.strftime("[%d.%m.%Y %H:%M:%S]: "), msg, file=stderr)
        file.write(time.strftime("[%d.%m.%Y %H:%M:%S]: ") + msg + os.linesep)

def log_to_csv(log_path_folder):
    for i in range(8):
        log_path = os.path.join(log_path_folder,str(i)+"_log.txt")
        csv_path = os.path.join(log_path_folder,str(i)+"_log.csv")
        csv =[]
        csv_line = []

        f = open(log_path, "r")
        lines = f.readlines()
        for line in lines:
            x = line.split()
            if x[2] == "Epoch":
                epoch_num = int((x[3][:-1]))
            if x[2] == "Overall:":
                csv_line.append(epoch_num)

                Acc = float(x[5])
                csv_line.append(Acc)
                Avg_F1 = float(x[8])
                csv_line.append(Avg_F1)
                Edit = float(x[11])
                csv_line.append(Edit)
                F1_10 = float(x[13])
                csv_line.append(F1_10)
                F1_25 = float(x[15])
                csv_line.append(F1_25)
                F1_50 =  float(x[17])
                csv_line.append(F1_50)
                csv.append(csv_line)
                csv_line = []
        headers= ["Epoch Num", "Acc", "Avg_F1", "Edit", "F1_10", "F1_25", "F1_50"]

        df = pd.DataFrame(csv, columns =headers)
        df.to_csv(csv_path,index=False)

    print(df)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, loss=False):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if not loss: # if calculating accuracy
            self.avg = self.avg * 100.0 # to percent


def reg_l2(model,l2_lambda,device ):
    l2_reg = torch.tensor(0.).to(device)
    for param in model.parameters():
        l2_reg += torch.norm(param)
    return l2_lambda * l2_reg


if __name__ == '__main__':
    log_to_csv("cross_valid results_with_cuDNN")
