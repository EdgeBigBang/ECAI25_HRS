for data_file in ECW08.csv; do
  for target in mps; do
    for len in 24; do
      task_name="${data_file%%.*}_${len}"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --model "HRS" \
        --seq_len "$len" \
        --pred_len "$len" \
        --token_mlp_dim 256 \
        --hidden_dim 16 \
        --h 32 \
        --expand 1 \
        --channel 1 \
        --lw 3 \
        --dropout 0.05 \
        --inverse 1 \
        --n_blocks 11 \
        --freq "h" \
        --patch_size 8 8 \
        --stride 6 6 \
        --dimension_mlp_dim 16 \
        --target "$target" \
        --is_training 1\
        --inverse 1 \
        --patience 10 \
        --learning_rate 0.01 \
        --draw_test 0 \
        --lradj "optim" \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done
