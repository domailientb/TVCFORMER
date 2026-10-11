# Kết quả TVCFormer - ETTm1

| Pred_len | MSE        | MAE        | RSE        |
|----------|------------|------------|------------|
| 96       | 0.28850044 | 0.34578973 | 0.52057844 |
| 192      | 0.33161572 | 0.36587685 | 0.54772425 |
| 336      | 0.36226815 | 0.38391072 | 0.57267922 |
| 720      | 0.41697350 | 0.41518873 | 0.61424464 |

- Dataset: ETTm1
- Seq_len: 336
- Device: GPU (`cuda:0`)
- Batch size: 128
- Early stopping: Yes
- Config: `official/configs/ETTm1.yaml`
