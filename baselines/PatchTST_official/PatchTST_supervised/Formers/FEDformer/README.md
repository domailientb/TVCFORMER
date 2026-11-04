# FEDformer
From https://github.com/MAZiqing/FEDformer


Frequency Enhanced Decomposed
Transformer (FEDformer) is more efficient than
standard Transformer with a linear complexity
to the sequence length. 

Our empirical studies
with six benchmark datasets show that compared
with state-of-the-art methods, FEDformer can
reduce prediction error by 14.8% and 22.6%
for multivariate and univariate time series,
respectively.

## Ghi chú tiếng Việt

- Thư mục này chứa mã FEDformer tham khảo, là baseline riêng chứ không phải phần triển khai PatchTST.
- FEDformer kết hợp phân rã chuỗi với xử lý trong miền tần số để giảm chi phí attention khi chuỗi dài.
- Các hướng dẫn chạy bên dưới thuộc môi trường và dữ liệu của bản FEDformer gốc; cần kiểm tra đường dẫn dữ liệu, phiên bản thư viện và script trước khi chạy trong dự án này.
- `scripts/` chứa cấu hình lệnh thí nghiệm; `layers/`, `models/` và `exp/` lần lượt chứa các phép biến đổi, kiến trúc và vòng lặp thí nghiệm.


## Get Started

1. Install Python 3.6, PyTorch 1.9.0.
2. Download data. You can obtain all the six benchmarks from xxxx.
3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/run_M.sh
bash ./scripts/run_S.sh
```


## Citation

If you find this repo useful, please cite our paper. 

```
xxxxx
```

## Contact

If you have any question or want to use the code, please contact xxx@xxxx .

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data
