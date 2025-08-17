for data_file in ECW08.csv; do
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
        --h 24 \
        --lw 3 \
        --expand 1 \
        --channel 1 \
        --hidden_dim 64 \
        --dimension_mlp_dim 128 \
        --patch_size 36 10 \
        --stride 1 1 \
        --token_mlp_dim 1024 \
        --n_blocks 11 \
        --dropout 0 \
        --learning_rate 0.001 \
        --freq "h" \
        --target "$target" \
        --is_training 1 \
        --inverse 1 \
        --patience 10 \
        --draw_test 0 \
        --lradj "optim" \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done
