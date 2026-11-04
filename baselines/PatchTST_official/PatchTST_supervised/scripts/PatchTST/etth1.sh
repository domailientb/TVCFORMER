# Ghi chú (VI): Script chạy thí nghiệm PatchTST trên cấu hình/bộ dữ liệu ETTH1.
# Cấu hình đáng chú ý: seq_len=336, model_name=PatchTST, root_path_name=./dataset/,
#    data_path_name=ETTh1.csv, pred_len=96 192 336 720.

# Kiểm tra và tạo thư mục logs nếu chưa tồn tại.
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Tạo thư mục lưu log cho các thí nghiệm dự báo dài hạn.
if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Khai báo độ dài chuỗi đầu vào và mô hình PatchTST.
seq_len=336
model_name=PatchTST

# Khai báo đường dẫn dữ liệu, tên mô hình và tên bộ dữ liệu.
root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

# Thiết lập random seed để đảm bảo tính nhất quán giữa các lần chạy.
random_seed=2021

# Lặp qua bốn độ dài dự báo: 96, 192, 336 và 720 bước thời gian.
for pred_len in 96 192 336 720
do
    # Khởi chạy thí nghiệm huấn luyện và đánh giá PatchTST.
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done