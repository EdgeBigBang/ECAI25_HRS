for data_file in ECW09.csv; do
  for target in mps; do
    for len in 72; do
      task_name="${data_file%%.*}_${len}"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --model "HRS" \
        --seq_len "$len" \
        --pred_len "$len" \
        --token_mlp_dim 32 \
        --hidden_dim 32 \
        --h 24 \
        --freq "h" \
        --patch_size 8 8 \
        --stride 4 4 \
        --dimension_mlp_dim 32 \
        --target "$target" \
        --freq "h" \
        --is_training 1 \
        --inverse 1 \
        --patience 10 \
        --learning_rate 0.001 \
        --lradj "type1" \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done
