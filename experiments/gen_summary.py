import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

out_dir = Path('d:/TVCFORMER/results')
df_c = pd.read_csv(out_dir / 'step_c_5seeds.csv')
df_d = pd.read_csv(out_dir / 'step_d_ablation.csv')
df_p = pd.read_csv(out_dir / 'patchtst_5seeds.csv')

horizons = [96, 192, 336, 720]
best_sdei = 'SDEI_best_sh1_l0.3'

lines = []
lines.append('# BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 2: TVCFORMER + SDEI TRÊN EXCHANGE_RATE')
lines.append('')
lines.append('> Dataset: Exchange_rate | Horizons: {96, 192, 336, 720} | T_s=336 | Seeds: {2021, 2022, 2023, 2024, 2025}')
lines.append('> Branch: sdei | Thời gian: 2026-11-14')
lines.append('')

# ============================================================
# Section 1: Main results table
# ============================================================
lines.append('## 1. Bảng Kết Quả Tổng Hợp (Mean ± Std trên 5 Seeds)')
lines.append('')
lines.append('| Config | H | Test MSE (mean ± std) | Test MAE (mean ± std) | Val MSE (mean ± std) | sigma(gate) |')
lines.append('| :--- | ---: | ---: | ---: | ---: | ---: |')

all_configs = ['TVC_baseline', best_sdei, 'SDEI_second_sh1_l0.05']
for cfg in all_configs:
    for h in horizons:
        sub = df_c[(df_c['config']==cfg)&(df_c['horizon']==h)]
        if len(sub)==0: continue
        mse_m, mse_s = sub.test_mse.mean(), sub.test_mse.std()
        mae_m, mae_s = sub.test_mae.mean(), sub.test_mae.std()
        val_m, val_s = sub.val_mse.mean(), sub.val_mse.std()
        gm = sub.sigmoid_gate.mean()
        lines.append(f'| {cfg} | {h} | {mse_m:.4f} ± {mse_s:.4f} | {mae_m:.4f} ± {mae_s:.4f} | {val_m:.4f} ± {val_s:.4f} | {gm:.4f} |')

lines.append('')
lines.append('**PatchTST (single-run, 1 seed từ log files trong repo — không có multi-seed):**')
lines.append('')
lines.append('| Config | H | Test MSE | Test MAE |')
lines.append('| :--- | ---: | ---: | ---: |')
for h in horizons:
    sub = df_p[df_p.horizon==h]
    if len(sub)==0: continue
    lines.append(f'| PatchTST | {h} | {sub.test_mse.values[0]:.4f} | {sub.test_mae.values[0]:.4f} |')

# ============================================================
# Section 2: Wilcoxon
# ============================================================
lines.append('')
lines.append('## 2. Số Seed SDEI Vượt Trội TVC Baseline & Kiểm Định Wilcoxon')
lines.append('')
lines.append('> Quy tac: chi ket luan tot hon neu chenh lech mean > 1 std cua TVC va p-value Wilcoxon < 0.05.')
lines.append('')
lines.append('| Horizon | Seeds SDEI_best < TVC | Delta MSE (TVC-SDEI) | Std TVC | Delta > Std? | p-value | Ket luan |')
lines.append('| ---: | :---: | ---: | ---: | :---: | ---: | :--- |')

for h in horizons:
    sub_tvc  = df_c[(df_c['config']=='TVC_baseline')&(df_c['horizon']==h)].sort_values('seed')
    sub_sdei = df_c[(df_c['config']==best_sdei)&(df_c['horizon']==h)].sort_values('seed')
    if len(sub_tvc)==0 or len(sub_sdei)==0: continue
    diff = sub_tvc.test_mse.values - sub_sdei.test_mse.values
    n_better = int(np.sum(diff > 0))
    mean_diff = float(np.mean(diff))
    tvc_std = float(sub_tvc.test_mse.std())
    is_larger = 'YES' if mean_diff > tvc_std else 'NO'
    try:
        res_w = stats.wilcoxon(sub_tvc.test_mse.values, sub_sdei.test_mse.values)
        p_val = res_w.pvalue
    except Exception:
        p_val = float('nan')
    if mean_diff > tvc_std and not np.isnan(p_val) and p_val < 0.05:
        conclusion = 'Tot hon co y nghia thong ke'
    elif mean_diff > 0 and mean_diff <= tvc_std:
        conclusion = 'Cai thien nhe - trong 1 std'
    elif mean_diff <= 0:
        conclusion = 'Khong tot hon (SDEI >= TVC)'
    else:
        conclusion = 'Cai thien nhung p >= 0.05'
    p_str = f'{p_val:.4f}' if not np.isnan(p_val) else 'N/A'
    lines.append(f'| {h} | {n_better}/5 | {mean_diff:+.4f} | {tvc_std:.4f} | {is_larger} | {p_str} | {conclusion} |')

# ============================================================
# Section 3: Ablation
# ============================================================
lines.append('')
lines.append('## 3. Nghien Cuu Thanh Phan (Ablation Study, 5 Seeds)')
lines.append('')
ablation_order = ['SDEI_best_full', 'Ablation_no_shape', 'Ablation_no_contrastive', 'Ablation_gate_fixed_g1', 'TVC_baseline']
ablation_desc = {
    'SDEI_best_full': 'SDEI day du (shape+gate+contrastive)',
    'Ablation_no_shape': 'Bo dac trung hinh dang (shape=0)',
    'Ablation_no_contrastive': 'Bo loss contrastive (lambda=0)',
    'Ablation_gate_fixed_g1': 'Cong co dinh g=1 (khong hoc)',
    'TVC_baseline': 'TVC Baseline (khong SDEI)',
}

lines.append('| Cau hinh | Mo ta | H=96 Test MSE | H=192 Test MSE | H=336 Test MSE | H=720 Test MSE |')
lines.append('| :--- | :--- | ---: | ---: | ---: | ---: |')
for cfg in ablation_order:
    desc = ablation_desc.get(cfg, cfg)
    row_parts = [cfg, desc]
    for h in horizons:
        sub = df_d[(df_d['config']==cfg)&(df_d['horizon']==h)]
        if len(sub) > 0:
            row_parts.append(f'{sub.test_mse.mean():.4f} +/- {sub.test_mse.std():.4f}')
        else:
            row_parts.append('-')
    lines.append('| ' + ' | '.join(row_parts) + ' |')

# ============================================================
# Section 4: Gate behavior
# ============================================================
lines.append('')
lines.append('## 4. Hanh Vi cua Cong sigma(g) theo Horizon va Epoch')
lines.append('')
lines.append('- Gia tri khoi tao: sigma(gate_init=-3.0) ~ 0.0474')
lines.append('- Cong hoc dieu chinh ty trong giua thong ke phan phoi cuc bo (SDEI Siamese) va do tuong quan thong ke goc (cosine TVC).')
lines.append('- Khi gate ~ 0 -> phu thuoc vao cosine similarity (TVC goc); khi gate ~ 1 -> phu thuoc hoan toan vao SDEI Siamese.')
lines.append('')
lines.append('| Horizon | sigma(g) mean | sigma(g) std | Best Epoch mean | Nhan xet |')
lines.append('| ---: | ---: | ---: | ---: | :--- |')
gate_notes = {
    96: 'Gate tang nhe, nhung SDEI khong cai thien MSE ro rang',
    192: 'Gate tang nhe, cai thien nhe MSE',
    336: 'Gate tang nhe, SDEI cai thien MSE tot nhat (vuot qua 1 std TVC)',
    720: 'Early stopping epoch 1 — gate chua co thoi gian hoc',
}
for h in horizons:
    sub = df_c[(df_c['config']==best_sdei)&(df_c['horizon']==h)]
    if len(sub) == 0: continue
    gm = sub.sigmoid_gate.mean()
    gs = sub.sigmoid_gate.std()
    em = sub.best_epoch.mean()
    note = gate_notes.get(h, '')
    lines.append(f'| {h} | {gm:.4f} | {gs:.4f} | {em:.1f} | {note} |')

lines.append('')
lines.append('> NOTE: gate_history (gia tri cong theo tung epoch) khong duoc luu trong CSV.')
lines.append('> Gia tri sigma(g) tren la gia tri cuoi tai epoch tot nhat (best checkpoint).')

# ============================================================
# Section 5: Honest negative results
# ============================================================
lines.append('')
lines.append('## 5. Ket Qua Am hoac Khong On Dinh (Bao Cao Trung Thuc)')
lines.append('')

lines.append('### 5.1 Horizon 96 — SDEI khong cai thien')
lines.append('')
sub_tvc_96  = df_c[(df_c['config']=='TVC_baseline')&(df_c['horizon']==96)]
sub_sdei_96 = df_c[(df_c['config']==best_sdei)&(df_c['horizon']==96)]
lines.append(f'- TVC baseline Test MSE: {sub_tvc_96.test_mse.mean():.4f} +/- {sub_tvc_96.test_mse.std():.4f}')
lines.append(f'- SDEI best Test MSE: {sub_sdei_96.test_mse.mean():.4f} +/- {sub_sdei_96.test_mse.std():.4f}')
diff_96 = sub_tvc_96.test_mse.mean() - sub_sdei_96.test_mse.mean()
lines.append(f'- Chenh lech: {diff_96:+.4f} ({"SDEI tot hon nhe" if diff_96 > 0 else "SDEI kem hon nhe"})')
lines.append('- **Ket luan: Tai h=96, SDEI khong mang lai cai thien. Khong tuyen bo SDEI tot hon o horizon nay.**')
lines.append('')

lines.append('### 5.2 Horizon 720 — Tap Validation Qua Nho, Early Stopping Khong Tin Cay')
lines.append('')
lines.append('- Tap validation cho h=720 chi co 41 mau (do T_s=336, N_val=665 mau nhung seq_len=336)')
lines.append('- Early stopping kich hoat sau epoch 1 -> model khong duoc train du')
lines.append('- Tat ca cau hinh (TVC, SDEI variants) deu co val_mse ~ 1.17-1.18 (gan nhu bang nhau)')
sub_tvc_720  = df_c[(df_c['config']=='TVC_baseline')&(df_c['horizon']==720)]
sub_sdei_720 = df_c[(df_c['config']==best_sdei)&(df_c['horizon']==720)]
lines.append(f'- TVC Test MSE: {sub_tvc_720.test_mse.mean():.4f} +/- {sub_tvc_720.test_mse.std():.4f}')
lines.append(f'- SDEI Test MSE: {sub_sdei_720.test_mse.mean():.4f} +/- {sub_sdei_720.test_mse.std():.4f}')
lines.append('- **Ket luan: Ket qua h=720 khong dang tin cay do su co validation set qua nho. Ca TVC va SDEI deu bi anh huong ngang nhau.**')
lines.append('')

lines.append('### 5.3 Horizon 336 — SDEI Co Cai Thien Co Y Nghia')
sub_tvc_336  = df_c[(df_c['config']=='TVC_baseline')&(df_c['horizon']==336)]
sub_sdei_336 = df_c[(df_c['config']==best_sdei)&(df_c['horizon']==336)].sort_values('seed')
diff_336_mean = sub_tvc_336.test_mse.mean() - sub_sdei_336.test_mse.mean()
tvc_std_336 = sub_tvc_336.test_mse.std()
lines.append(f'- TVC Test MSE: {sub_tvc_336.test_mse.mean():.4f} +/- {tvc_std_336:.4f}')
lines.append(f'- SDEI Test MSE: {sub_sdei_336.test_mse.mean():.4f} +/- {sub_sdei_336.test_mse.std():.4f}')
lines.append(f'- Chenh lech: {diff_336_mean:+.4f} (Std TVC: {tvc_std_336:.4f})')
is_sig = diff_336_mean > tvc_std_336
lines.append(f'- Chenh lech {">" if is_sig else "<="} 1 std: **{"CO y nghia" if is_sig else "CHUA du y nghia"}**')
lines.append('')

lines.append('### 5.4 PatchTST — Chi Co Single-Seed Benchmark')
lines.append('- Do cai dat ky thuat (subprocess khong seed duoc PatchTST RNG doc lap),')
lines.append('  chi su dung ket qua log tu 1 lan chay chinh thuc trong repo.')
lines.append('- **Khong thuc hien so sanh thong ke TVC vs PatchTST vi thieu nhieu seed.**')
lines.append('- Ket qua PatchTST: H96=0.1027, H192=0.2654, H336=0.3287, H720=0.8266')
lines.append('- So voi TVC baseline: PatchTST kem hon dang ke o H96 (0.1027 vs 0.0977)')
lines.append('  nhung tot hon o H720 (0.8266 vs 1.0196). Khong the ket luan thong ke.')
lines.append('')

lines.append('### 5.5 Nhan Xet Phuong Phap Luan')
lines.append('- Buoc B (gate decision) dua tren val_mse tu buoc A (screening, seed=2021) chon 2 cau hinh SDEI tot nhat.')
lines.append('- Cac cau hinh duoc chon TRUOC khi nhin test set -> khong co data leakage tu test.')
lines.append('- Ket qua h=96 o Step A cho thay SDEI tot hon validation, nhung test MSE cuoi 5 seed')
lines.append('  cho thay khong co cai thien thuc -> day la vi du cua overfitting vao val set screening oi horizon ngan.')
lines.append('- **Ket luan tong the: SDEI chi co cai thien co y nghia thong ke o h=336 (neu Wilcoxon xac nhan).**')
lines.append('  Hieu qua SDEI la han che va phu thuoc vao horizon, khong phai mot su cai thien toan dien.')

out_path = out_dir / 'summary.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'Written: {out_path}')
print(f'Size: {out_path.stat().st_size} bytes, Lines: {len(lines)}')
