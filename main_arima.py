import argparse
import random
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from load_data import *
from sensors2graph import *
from utils import *
from tqdm import tqdm
from logger import Logger
from torch.utils.data import DataLoader
from sklearn.svm import SVR
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="STGCN_WAVE")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="batch size for training and validation (default: 64)",
)
parser.add_argument(
    "--epochs", type=int, default=200, help="epochs for training  (default: 200)"
)
parser.add_argument(
    "--num_layers", type=int, default=1, help="number of layers"
)
parser.add_argument(
    "--nheads", type=int, default=1, help="number of att heads"
)
parser.add_argument(
    "--tree_num_layers", type=int, default=10, help="number of tree layers"
)
parser.add_argument("--window", type=int, default=12, help="window length")

parser.add_argument('--filename', type=str, default='pems04')
parser.add_argument(
    "--sensorsfilepath",
    type=str,
    default="./data/PEMS04/graph_sensor_ids.txt",
    help="sensors file path",
)
parser.add_argument(
    "--disfilepath",
    type=str,
    default="./data/PEMS04/PEMS04.csv",
    help="distance file path",
)
parser.add_argument(
    "--savemodelpath",
    type=str,
    default="./best_model/PEMS04/arima.pt",
    help="save model path",
)
parser.add_argument(
    "--pred_len",
    type=int,
    default=12,
    help="how many steps away we want to predict",
)
parser.add_argument(
    "--control_str",
    type=str,
    default="TSTN",
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
parser.add_argument(
    "--channels",
    type=int,
    nargs="+",
    default=[1, 32, 32, 64, 64, 128],
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
args = parser.parse_args()

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.disablecuda
    else torch.device("cpu")
)

with open(args.sensorsfilepath) as f:
    sensor_ids = f.read().strip().split(",")

distance_df = pd.read_csv(args.disfilepath, dtype={"from": "str", "to": "str"})

adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
sp_mx = sp.coo_matrix(adj_mx)


# read data
# df = pd.read_hdf(args.tsfilepath)
# num_samples, num_nodes = df.shape
# tsdata = df.to_numpy()


data_set = Data_load(args)

save_path = args.savemodelpath
n_his = args.window
n_pred = args.pred_len
blocks = args.channels
drop_prob = 0
num_layers = args.num_layers
tree_num_layers = args.tree_num_layers

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr
W = adj_mx


# generate data_loader

train_data = torch.utils.data.TensorDataset(data_set['train_input'], data_set['train_target'])
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(data_set['eval_input'], data_set['eval_target'])
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(data_set['test_input'], data_set['test_target'])
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


num_nodes = data_set['num_nodes']
input_features = data_set['input_features']


loss = nn.MSELoss()
# x_train (b, n, t, c) y_train(b, n, t)
x_train, y_train = data_set['train_input'], data_set['train_target']
x_val, y_val = data_set['eval_input'], data_set['eval_target']
x_test, y_test = data_set['test_input'],  data_set['test_target']

time, timesteps = y_test.shape[0], y_test.shape[2]

x_train_cp = x_train
y_test_cp = y_test
# x shape is (t_all, n, t_12)
x_train = x_train[:, :, :, 0].view(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
y_train = y_train.view(y_train.shape[0], y_train.shape[1] * y_train.shape[2])
x_val = x_val[:, :, :, 0].view(x_val.shape[0], x_val.shape[1] * x_val.shape[2])
y_val = y_val.view(y_val.shape[0], y_val.shape[1] * y_val.shape[2])
x_test = x_test[:, :, :, 0].view(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
y_test = y_test.view(y_test.shape[0], y_test.shape[1] * y_test.shape[2])


# 平稳性检测
# 单位根检验-ADF检验
print(sm.tsa.stattools.adfuller(x_train[:, 0]))

# 白噪声检测
print(acorr_ljungbox(x_train[:, 0], lags=[6, 12], boxpierce=True))

# 计算ACF,PACF
# acf = plot_acf(x_train[:, 0])
# plt.title("node 1 self-cor figure")
# plt.show()
# # plt.savefig(r"D:/mzbfile/2024年寒假/20231117注意力树/acf.png")
#
#
# pacf = plot_pacf(x_train[:, 0])
# plt.title("node 1 bias self-cor figure")
# plt.show()
# plt.savefig(r"D:/mzbfile/2024年寒假/20231117注意力树/pacf.png")

#  确定介数
# trend_evaluate = sm.tsa.arma_order_select_ic(x_train[:, 0], ic=['aic', 'bic'], trend='n', max_ar=20, max_ma=5)
# print('train AIC', trend_evaluate.aic_min_order)
# print('train BIC', trend_evaluate.bic_min_order)



len = x_train.shape[1]
y_pred_all = []

# train
# train = y_test[:, 12].tolist()
train = x_train_cp[:, 12, 0, 0].tolist()
model = sm.tsa.arima.ARIMA(train, order=(7, 0, 4))
arima_res = model.fit()
arima_res.summary()

# with open(result_path + f"arima.pkl", 'wb') as f:
#     pickle.dump(arima_res, f)
test_shape1 = y_test.shape[0]
test_shape2 = y_test.shape[1]
y_res = arima_res.predict(0, test_shape1-1)
# for i in range(len):
#     y_res = arima_res.predict(0, test_shape1-1)
#     y_pred_all.append(y_res)

# y_pred_all = torch.from_numpy(np.array(y_pred_all)).transpose(1, 0)

# y_pred_all = y_pred_all.repeat(1, 2484)

# get MAE, RMSE, MAPE

mae = []
mape = []
mse = []

std = data_set['data_std']
mean = data_set['data_mean']

y_test = y_test_cp[:, 12, 0].cpu().numpy().reshape(-1)
y_pred_all = y_res.astype(np.float32)
# y_pred_all = y_pred_all.cpu().numpy().reshape(-1)

# inverse normalization
y_test, y_pred_all = Un_Z_Score(y_test, mean, std), Un_Z_Score(y_pred_all, mean, std)

d = np.abs(y_test-y_pred_all)

mae += d.tolist()
mape += (d / y_test).tolist()
mse += (d**2).tolist()
MAE = np.array(mae).mean()
MAPE = np.array(mape).mean()
RMSE = np.sqrt(np.array(mse).mean())

elogger = Logger('run_log_arima')

print("test result:", "MAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
elogger.log(f"test result: MAE:{MAE}, MAPE:{MAPE}, RMSE:{RMSE}")
elogger.log("######################")


# save result
# y_pred = y_pred_all.reshape(time, num_nodes, timesteps)
# y_target = y_test.reshape(time, num_nodes, timesteps)

save_result_path = "result/PEMS04/arima/"
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)
np.savetxt(save_result_path + "pred_arima" + ".csv", y_res, delimiter=',')
np.savetxt(save_result_path + "true_arima" + ".csv", y_pred_all, delimiter=',')

