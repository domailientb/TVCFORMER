"""Training/evaluation adapter; keeps the official data and forecast protocol."""
import os
import sys
import time
import numpy as np
import torch
from torch.optim import lr_scheduler
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'official'
SIAMESE_DIR = Path(__file__).resolve().parent
if str(SIAMESE_DIR) in sys.path:
    sys.path.remove(str(SIAMESE_DIR))
if str(OFFICIAL) in sys.path:
    sys.path.remove(str(OFFICIAL))
sys.path.insert(0, str(OFFICIAL))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from exp.exp_main import Exp_Main


class Exp_Siamese(Exp_Main):
    def _build_model(self):
        from FinalProject.Siamese_TVC.models.tvcformer_siamese import Model
        return Model(self.args).float().to(self.device)

    def train(self, setting):
        # Match the official optimizer/scheduler/early-stopping protocol while
        # selecting checkpoints on validation only (never test each epoch).
        from utils.tools import EarlyStopping, adjust_learning_rate
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        self.best_validation_loss = float('inf')
        scheduler = lr_scheduler.OneCycleLR(
            optimizer, steps_per_epoch=len(train_loader),
            pct_start=self.args.pct_start, epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate)
        amp = bool(self.args.use_amp and self.device.type == 'cuda')
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.best_epoch = 1
        for epoch in range(self.args.train_epochs):
            self.model.train()
            losses = []
            started = time.time()
            for batch_x, batch_y, _, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp):
                    pred = self.model(batch_x)
                    truth = batch_y[:, -self.args.pred_len:, :]
                    loss = criterion(pred, truth)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().item()))
                if self.args.lradj == 'TST':
                    scheduler.step()
            val_loss = self.vali(vali_data, vali_loader, criterion)
            gate = getattr(self.model, 'gate_value', None)
            gate_text = f' gate={float(gate):.4f}' if gate is not None else ''
            print(f'Epoch {epoch + 1}: train={np.mean(losses):.7f} '
                  f'val={val_loss:.7f}{gate_text} time={time.time()-started:.1f}s')
            stopping(val_loss, self.model, path)
            if stopping.counter == 0:
                self.best_epoch = epoch + 1
                self.best_validation_loss = float(val_loss)
            if stopping.early_stop:
                print('Early stopping')
                break
            if self.args.lradj != 'TST':
                adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)
        self.model.load_state_dict(torch.load(
            os.path.join(path, 'checkpoint.pth'), map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        _, loader = self._get_data(flag='test')
        checkpoint = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if test:
            self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()
        predictions, truths = [], []
        baseline_graphs = {}
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        if not hasattr(base_model, 'graph_diagnostics'):
            graph_modules = [m for m in base_model.modules()
                             if m.__class__.__name__ == 'graph_generator']
            def capture_graph(module, inputs, output):
                baseline_graphs['adjacency'] = output[2].detach().coalesce()
                baseline_graphs['edge_index'] = inputs[1].detach()
            handles = [m.register_forward_hook(capture_graph) for m in graph_modules]
        else:
            handles = []
        with torch.no_grad():
            for batch_x, batch_y, _, _ in loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.detach().cpu().numpy())
                truths.append(batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy())
        for handle in handles:
            handle.remove()
        snapshots = base_model.graph_diagnostics() if hasattr(base_model, 'graph_diagnostics') else []
        if snapshots:
            snap = snapshots[0]
            graph = snap['adjacency'].coalesce()
            os.makedirs(self.args.results_dir, exist_ok=True)
            np.savez(os.path.join(self.args.results_dir, setting + '_graph.npz'),
                     graph_indices=graph.indices().cpu().numpy(),
                     graph_values=graph.values().cpu().numpy(),
                     graph_shape=np.asarray(graph.shape),
                     edge_index=snap['edge_index'].cpu().numpy(),
                     embeddings=snap['embeddings'].cpu().numpy(),
                     pearson=snap['pearson'].cpu().numpy(),
                     **{f'candidate_{key}': value.cpu().numpy()
                        for key, value in snap['candidate_diagnostics'].items()},
                     gate=np.asarray(float(snap['gate'].cpu())),
                     graph_build_seconds=np.asarray(snap['graph_build_seconds']))
        elif baseline_graphs:
            graph = baseline_graphs['adjacency']
            os.makedirs(self.args.results_dir, exist_ok=True)
            np.savez(os.path.join(self.args.results_dir, setting + '_graph.npz'),
                     graph_indices=graph.indices().cpu().numpy(),
                     graph_values=graph.values().cpu().numpy(),
                     graph_shape=np.asarray(graph.shape),
                     edge_index=baseline_graphs['edge_index'].cpu().numpy(),
                     gate=np.asarray(0.0), graph_build_seconds=np.asarray(np.nan))
        predictions, truths = np.concatenate(predictions), np.concatenate(truths)
        from utils.metrics import metric
        mae, mse, rmse, mape, mspe, rse, corr = metric(predictions, truths)
        os.makedirs(self.args.results_dir, exist_ok=True)
        np.savez(os.path.join(self.args.results_dir, setting + '.npz'),
                 pred=predictions, true=truths)
        csv_path = os.path.join(self.args.results_dir, 'metrics.csv')
        with open(csv_path, 'a', encoding='utf-8') as f:
            if f.tell() == 0:
                f.write('setting,mse,mae,rmse,mape,mspe,rse,corr\n')
            f.write(f'{setting},{mse},{mae},{rmse},{mape},{mspe},{rse},{corr}\n')
        print(f'Test MSE={mse:.8f} MAE={mae:.8f}')
        return mse, mae
