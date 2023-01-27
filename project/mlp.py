# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os,glob
import pandas as pd
from scipy import interpolate

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure




# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'Adagrad',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'weight_decay':0.01,        # normalize
        #'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 600,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth', # your model will be saved here
    'train_data_path':"./mlp_data" ,
    'test_data_path':"./data",
}

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()


def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

#dataset
    class AngleDataset(Dataset):
        ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,path,mode='train',target_only=False):
        #命名规范：名字后面加1


        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


class AngleDataset(Dataset):
    ''' Dataset for loading and preprocessing the dataset,path is data_folder name '''
    def __init__(self,folder,mode='train',target_only=False):
        '''要求：命名规范：Accelerometer_1.csv'''
        #TODO 突然意识到这个文件顺序我有给保证吗？
        self.path = os.path.join(os.getcwd(),folder)

        Accel_files = glob.glob(os.path.join(self.path, "Accelerometer*.csv"))
        df_acc = [pd.read_csv(f).values for f in Accel_files]

        Gyr_files = glob.glob(os.path.join(self.path, "Gyroscope*.csv"))
        df_gyr = [pd.read_csv(f).values for f in Gyr_files]
        
        Lin_files = glob.glob(os.path.join(self.path, "Linear Accelerometer*.csv"))
        df_lin = [pd.read_csv(f).values for f in Lin_files]

        Mag_files = glob.glob(os.path.join(self.path, "Magnetometer*.csv"))
        df_mag = [pd.read_csv(f).values for f in Mag_files]

        num_frame = len(Gyr_files)
        norm_len = [min([len(df_acc[i]),len(df_gyr[i]),len(df_lin[i]),len(df_mag[i]),]) for i in range(num_frame)]
        
        for i in range(num_frame):
            df_acc[i] = df_acc[i][:norm_len[i],:]
            df_gyr[i] = df_gyr[i][:norm_len[i],:]
            df_lin[i] = df_lin[i][:norm_len[i],:]
            df_mag[i] = df_mag[i][:norm_len[i],:]


        if num_frame > 1:
            acc = np.concatenate(df_acc, axis=0)
            gyr = np.concatenate(df_gyr, axis=0)
            lin = np.concatenate(df_lin, axis=0)
            mag = np.concatenate(df_mag, axis=0)
        else:
          acc= np.array(df_acc[0])
          gyr = np.array(df_gyr[0])
          lin = np.array(df_lin[0])
          mag = np.array(df_mag[0])



        self.data_dic = {"acc":acc[:,1:].astype(np.float64), \
                "gyr": gyr[:,1:].astype(np.float64),\
                "lin": lin[:,1:].astype(np.float64),\
                "mag": mag[:,1:].astype(np.float64),\
                "time": gyr[:,0].astype(np.float64),
                    }

        if mode == 'test':
            f = os.path.join(self.path, "Location_input.csv")
            loc = pd.read_csv(f, sep=',')
            self.data_dic["loc"]= loc.values[:,[0,5]].astype(np.float64)
            

        else:
            Loc_files = glob.glob(os.path.join(self.path, "Location ans*.csv"))
            loc_mag = [pd.read_csv(f).values for f in Loc_files]
            start_index = 0
            hole_time = gyr[:,0]
            label =[]
            for i in range(len(loc_mag)):
                time = hole_time[start_index : start_index + norm_len[i]]
                start_index += norm_len[i]
                loc_mag_tmp = np.array(loc_mag[i][:,5],dtype=np.float64)
                tck = interpolate.interp1d(loc_mag[i][:,0],loc_mag_tmp,fill_value="extrapolate")
                tmp_ans = tck(time)
                label.append(tmp_ans)

            loc = np.concatenate(label, axis=0)
            self.data_dic["loc"]= loc.astype(np.float64)


        data = np.concatenate((self.data_dic['acc'], \
                                self.data_dic['gyr'], \
                                self.data_dic['lin'], \
                                self.data_dic['mag'],\
                                ),
                                axis=1)

        self.mode = mode

        if mode == 'test':
            self.data = torch.FloatTensor(data)
        else:
            target = np.array(loc)
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 0:] = \
            (self.data[:, 0:] - self.data[:, 0:].mean(dim=0, keepdim=True)) \
            / self.data[:, 0:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of Angle Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)

def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = AngleDataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader



class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        ans = self.net(x).squeeze(1)
        return ans

    def l1_regularization(self, l1_alpha):
        l1_loss = []
        for module in self.modules():
            if type(module) is nn.BatchNorm2d:
                l1_loss.append(torch.abs(module.weight).sum())
        return l1_alpha * sum(l1_loss)
 
    def l2_regularization(self, l2_alpha):
        l2_loss = []
        for module in self.modules():
            if type(module) is nn.Conv2d:
                l2_loss.append((module.weight ** 2).sum() / 2.0)
        return l2_alpha * sum(l2_loss)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L1/L2 regularization here
        ans = self.criterion(pred, target)
        return ans

def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 5000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        print("dev_mse:{}".format(dev_mse))

        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

#validate
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss



class Mlp_Model():
    def __init__(self,train_path = "mlp_data") -> None:
        self.mlp_model = NeuralNet(12)
        self.model_loss = None
        self.model_loss_record = None
        try:
            self.mlp_model.load_state_dict(torch.load(os.path.join(os.getcwd(),config['save_path']))) #试图寻找过去model，想训练新model请更新
        except:
            self.train_new(train_path)

    def train_new(self,data_path=config['train_data_path']):
        '''train new model'''
        device = get_device()                 # get the current available device ('cpu' or 'cuda')
        os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
        target_only = False                   # TODO: Using 40 states & 2 tested_positive features

        tr_set = prep_dataloader(data_path, 'train', config['batch_size'], target_only=target_only)
        dv_set = prep_dataloader(data_path, 'dev', config['batch_size'], target_only=target_only)

        self.mlp_model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
        self.model_loss, self.model_loss_record = train(tr_set, dv_set, self.mlp_model, config, device)

    def angle_test(self,tt_set):
        '''test angle of mlp'''
        device = get_device()
        model = self.mlp_model
        model.eval()                                # set model to evalutation mode
        preds = []
        for x in tt_set:                            # iterate through the dataloader
            x = x.to(device)                        # move data to device (cpu/cuda)
            with torch.no_grad():                   # disable gradient calculation
                pred = model(x)                     # forward pass (compute output)
                preds.append(pred.detach().cpu())   # collect prediction
        preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
        return preds

    def test(self,tt_path=config['test_data_path']):
        target_only = False                   # TODO: Using 40 states & 2 tested_positive features
        tt_set = prep_dataloader(tt_path,'test',config['batch_size'],target_only=target_only) 
        if self.mlp_model == None:
            print("didn't have a model, plz train a new one!")
            return
        return self.angle_test(tt_set=tt_set)
        
