for data_file in ECW08.csv; do
  for target in mps; do
    for len in 48; do
      task_name="${data_file%%.*}_${len}"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --model "HRS" \
        --seq_len "$len" \
        --pred_len "$len" \
        --h 32 \
        --lw 3.0 \
        --expand 1 \
        --channel 1 \
        --hidden_dim 32 \
        --dimension_mlp_dim 32 \
        --freq "h" \
        --patch_size 16 16 \
        --stride 2 2 \
        --token_mlp_dim 256 \
        --n_blocks 11 \
        --dropout 0.05 \
        --learning_rate 0.002 \
        --target "$target" \
        --is_training 1 \
        --lradj "type1" \
        --patience 10 \
        --draw_test 0 \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done
