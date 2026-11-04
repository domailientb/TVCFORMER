# Ghi chú (VI): Script chạy thí nghiệm Linear/DLinear trên cấu hình/bộ dữ liệu ETTH1.
# Cấu hình đáng chú ý: seq_len=336, model_name=DLinear.

# Kiểm tra và tạo thư mục logs nếu chưa tồn tại.
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Tạo thư mục lưu log cho các thí nghiệm dự báo dài hạn.
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Độ dài chuỗi đầu vào và tên mô hình được sử dụng.
seq_len=336
model_name=DLinear

# Chạy thí nghiệm với độ dài dự báo 96 bước thời gian.
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'96.log

# Chạy thí nghiệm với độ dài dự báo 192 bước thời gian.
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'192.log

# Chạy thí nghiệm với độ dài dự báo 336 bước thời gian.
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'336.log

# Chạy thí nghiệm với độ dài dự báo 720 bước thời gian.
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 >logs/LongForecasting/$model_name'_'Etth1_$seq_len'_'720.log