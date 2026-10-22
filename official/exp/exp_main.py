# -----------------------------------------------------------------------------
# This file is adapted from PatchTST (https://github.com/PatchTST/PatchTST)
# Original project: “PatchTST: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.”
# Original repository license: Apache-2.0
# Copyright (c) Nie et al.
#
# Modifications made by Yuan Feng et al., 2025:
# Perform workflow optimization for TVCFormer.
# Enable output of intermediate features.
# -----------------------------------------------------------------------------

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import TVCFormer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from utils.visual_utils import visualize_input, visualize_variable_corr, visualize_graph
from utils.correlation_utils import get_pearson_correlation_vector

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

input_features = []

variable_correlation_features = []

def hook_input_corr(module, fea_in, fea_out):
    
    input_features.append(fea_in[0][0].data.cpu().numpy())  # C × seq_len
    variable_correlation_feature = get_pearson_correlation_vector(fea_in[0][0].unsqueeze(0)).squeeze(0).data.cpu().numpy()  # [C^2]
    variable_correlation_features.append(variable_correlation_feature)
    
    return None

graph_outs = []

def hook_graph(module, fea_in, fea_out):
    
    graph = fea_out[-1].to_dense().detach().cpu().numpy()  # [BS*C*L × BS*C*L]
    graph_outs.append(graph)
    return None


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
    
    def _build_model(self):
        
        model_dict = {
            'TVCFormer': TVCFormer,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model
    
    def _get_data(self, flag):
        
        data_set, data_loader = data_provider(self.args, flag)
        
        return data_set, data_loader
    
    def _select_optimizer(self):
        
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
        return model_optim
    
    def _select_criterion(self):
        
        criterion_y = nn.MSELoss()
        
        return criterion_y
    
    def vali(self, vali_data, vali_loader, criterion_y, mode_flag='vail'):
        
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self.model(batch_x)
                
                else:
                    output_y = self.model(batch_x)
                
                f_dim = 0
                batch_y_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred_y = output_y.detach().cpu()
                
                true_y = batch_y_y.detach().cpu()
                
                loss = criterion_y(pred_y, true_y)

                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss
    
    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion_y = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self.model(batch_x)
                        
                        f_dim = 0
                        batch_y_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion_y(output_y, batch_y_y)
                        train_loss.append(loss.item())
                else:
                    output_y = self.model(batch_x)
                    
                    f_dim = 0
                    batch_y_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion_y(output_y, batch_y_y)
                    train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion_y=criterion_y, mode_flag='vail')
            test_loss = self.vali(test_data, test_loader, criterion_y=criterion_y, mode_flag='test')
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, test=0):
        
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        output_y = self.model(batch_x)
                
                else:
                    output_y = self.model(batch_x)
                
                f_dim = 0
                batch_y_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = output_y.detach().cpu().numpy()
                true = batch_y_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 2 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        
        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'pred.npy', preds)
        
        return

    
    def visualize(self, setting, num):

        channel_num = self.args.enc_in
        time_node_num = self.args.time_node_num
        total_node_num = channel_num * time_node_num

        test_data, test_loader = self._get_data(flag='test')
        test_loader_len = len(test_data)
        assert num <= test_loader_len, (
        f"Số mẫu cần trực quan hóa ({num}) vượt quá "
        f"số mẫu test ({test_loader_len})."
)

        print('loading model')
        checkpoint_path = os.path.join(
            self.args.checkpoints, setting, 'checkpoint.pth'
        )
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(self.model)

        writer_path = os.path.join(
            './visualize_results/' + setting, 'visual_data/'
        )
        os.makedirs(writer_path, exist_ok=True)

        # Select batches before inference to avoid storing every graph.
        sample_indices = np.sort(
            np.random.choice(test_loader_len, size=num, replace=False)
        )

        input_features.clear()
        variable_correlation_features.clear()
        graph_outs.clear()

        self.model.model.register_forward_hook(hook_input_corr)
        self.model.model.graph_generator.register_forward_hook(hook_graph)

        try:
            self.model.eval()
            with torch.no_grad():
                for index in sample_indices:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = test_loader.dataset.__getitem__(index)
                    batch_x = torch.as_tensor(batch_x).unsqueeze(0).float().to(self.device)

                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            self.model(batch_x)
                    else:
                        self.model(batch_x)

                    input_tmp = input_features[-1]
                    colors_list = visualize_input(input_tmp, index, writer_path)

                    variable_correlation_tmp = variable_correlation_features[-1]
                    visualize_variable_corr(
                        variable_correlation_tmp, channel_num, index, writer_path
                    )

                    graph_tmp = graph_outs[-1][:total_node_num, :total_node_num]
                    visualize_graph(graph_tmp, index, writer_path)

                    input_features.clear()
                    variable_correlation_features.clear()
                    graph_outs.clear()
        finally:
            self.model.model._forward_hooks.clear()
            self.model.model.graph_generator._forward_hooks.clear()

        return
