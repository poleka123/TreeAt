import argparse
import torch
import torch.nn as nn
from load_data import *
from baslines.astgcn.astgcn import ASTGCN
from baslines.astgcn.model_config import *
from sensors2graph import *
from utils import *
from tqdm import tqdm
from logger import Logger
from torch.utils.data import DataLoader
from astgcn_data_read import *
# import dgl

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

parser.add_argument('--filename', type=str, default='metr-la')
parser.add_argument(
    "--sensorsfilepath",
    type=str,
    default="./data/sensor_graph/graph_sensor_ids.txt",
    help="sensors file path",
)
parser.add_argument(
    "--disfilepath",
    type=str,
    default="./data/sensor_graph/distances_la_2012.csv",
    help="distance file path",
)
parser.add_argument(
    "--tsfilepath", type=str, default="./data/metr-la.h5", help="ts file path"
)



parser.add_argument(
    "--savemodelpath",
    type=str,
    default="./best_model/astgcn.pt",
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

is_availavle = torch.cuda.is_available()


with open(args.sensorsfilepath) as f:
    sensor_ids = f.read().strip().split(",")





distance_df = pd.read_csv(args.disfilepath, dtype={"from": "str", "to": "str"})

adj_mx = get_adjacency_matrix(distance_df, sensor_ids)

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


data_set = Data_load(args)

# astgcn dataset
# data_set = read_and_generate_dataset(filename='./data/Metr-la/metr-la.npz',
#                                      num_of_weeks=1,
#                                      num_of_days=1,
#                                      num_of_hours=1,
#                                      timesteps_input=n_his,
#                                      timesteps_output=n_pred)

num_nodes = data_set['num_nodes']
input_features = data_set['input_features']

# generate data_loader
train_data = torch.utils.data.TensorDataset(data_set['train_input'], data_set['train_target'])
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(data_set['eval_input'], data_set['eval_target'])
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(data_set['test_input'], data_set['test_target'])
test_iter = torch.utils.data.DataLoader(test_data, batch_size)



all_backbones = get_backbones(K=3, num_of_weeks=1, num_of_days=1, num_of_hours=1, num_of_vertices=num_nodes, adj_mx=adj_mx)


loss = nn.MSELoss()
points_per_hour = 12
num_of_weeks = 1
num_of_days = 1
num_of_hours = 1
num_of_timesteps = [[points_per_hour * num_of_weeks, points_per_hour],
                    [points_per_hour * num_of_days, points_per_hour],
                    [points_per_hour * num_of_hours, points_per_hour]]
model = ASTGCN(
    num_for_prediction=n_pred,
    all_backbones=all_backbones,
    num_of_vertices=num_nodes,
    num_of_features=input_features,
    num_of_timesteps=num_of_timesteps,
    _device=device
).to(device)

optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
min_val_mae = np.inf
min_test_mae = np.inf
if_save_true = False
# file_path = "result.txt"
# with open(file_path, 'w') as file:
#     file.write("Tree Train Process: \n")

elogger = Logger('run_log_astgcn1')

for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    std = torch.tensor(data_set['data_std']).to(device)
    mean = torch.tensor(data_set['data_mean']).to(device)
    for x, y in tqdm(train_iter):
        model.train()
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_pred = model([x.permute(0, 1, 3, 2)])
        y, y_pred = Un_Z_Score(y, mean, std), Un_Z_Score(y_pred, mean, std)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()

    #
    torch.cuda.empty_cache()
    # val data
    val_loss, val_mae = evaluate_model(model, loss, val_iter, mean, std, device)


    # test data
    MAE, MAPE, RMSE = evaluate_metric(model, test_iter, mean, std, device)
    test_loss, test_mae = evaluate_model(model, loss, test_iter, mean, std, device)

    # if MAE < min_test_mae:
    #     min_test_mae = MAE
    #     test_train = data_set['test_input'].to(device)
    #     test_target = data_set['test_target'].to(device)
    #     # test_pred = model(test_train)
    #     # astgcn
    #     test_pred = model([test_train.permute(0, 1, 3, 2)])
    #     test_pred, test_target = Un_Z_Score(test_pred, mean, std), Un_Z_Score(test_target, mean, std)
    #     if not os.path.exists("./best_model/"):
    #         os.makedirs("./best_model/")
    #     torch.save(model.state_dict(), save_path)
    #     save_result_path = "./result/astgcn/"
    #     if not os.path.exists(save_result_path):
    #         os.makedirs(save_result_path)
    #     test_pred = test_pred.detach().numpy()
    #     test_target = test_target.detach().numpy()
    #     np.savetxt(save_result_path + "pred_" + str(epoch) + ".csv", test_pred[:, :, 0], delimiter=',')
    #     if if_save_true == False:
    #         if_save_true = True
    #         np.savetxt(save_result_path + "true_" + ".csv", test_target[:, :, 0], delimiter=',')


    print("epoch: {}, train loss: {}, validation loss: {}\n".format(epoch, l_sum/n, val_loss))
    elogger.log("epoch: {}, train loss: {}, validation loss: {}\n".format(epoch, l_sum/n, val_loss))

    print("test loss:", test_loss, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
    elogger.log("test loss:{} \n MAE: {}, MAPE:{}, RMSE:{}".format(test_loss, MAE, MAPE, RMSE))
    elogger.log("######################")


std = torch.tensor(data_set['data_std']).to(device)
mean = torch.tensor(data_set['data_mean']).to(device)
# best_model = STGCN_WAVE(c=blocks,
#                    n=num_nodes,
#                    Lk=G,
#                    STree=STree,
#                    nheads=1,
#                    time_input=n_his,
#                    time_output=n_pred,
#                    control_str=args.control_str).to(device)

# best_model = GAT_WAVE(c=blocks,
#                       T=n_his,
#                       n=num_nodes,
#                       Lk=G,
#                       num_heads=1,
#                       control_str='SN').to(device)
best_model = ASTGCN(
    num_for_prediction=n_pred,
    all_backbones=all_backbones,
    num_of_vertices=num_nodes,
    num_of_features=input_features,
    num_of_timesteps=n_his,
    _device=device
).to(device)
best_model.load_state_dict(torch.load(save_path))
l = evaluate_model(best_model, loss, test_iter, mean, std, device)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, mean, std, device)
print("final test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
