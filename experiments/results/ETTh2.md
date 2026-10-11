# Kết quả TVCFormer - ETTh2

| Pred_len | MSE        | MAE        | RSE        |
|----------|------------|------------|------------|
| 96       | 0.27344292 | 0.33489630 | 0.41773763 |
| 192      | 0.33650035 | 0.37778836 | 0.46515137 |
| 336      | 0.32352257 | 0.38003573 | 0.45440143 |
| 720      | 0.37429574 | 0.41818500 | 0.49049509 |

- Dataset: ETTh2
- Seq_len: 336
- Device: GPU (`cuda:0`)
- Batch size:** 128
- onfig: `official/configs/ETTh2.yaml`
- Early stopping: yes
