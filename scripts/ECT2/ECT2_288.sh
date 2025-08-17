for data_file in ECT2.csv; do
  for target in mps; do
    for len in 288; do
      task_name="${data_file%%.*}_${len}_mul"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --seq_len "$len" \
        --model "HRS" \
        --pred_len "$len" \
        --lw 3 \
        --h 160 \
        --expand 1 \
        --channel 1 \
        --hidden_dim 128 \
        --dimension_mlp_dim 128 \
        --patch_size 16 16 \
        --stride 6 6 \
        --token_mlp_dim 256 \
        --n_blocks 10 \
        --dropout 0.05 \
        --learning_rate 0.003 \
        --freq "t" \
        --lradj "type1" \
        --patience 10 \
        --target "$target" \
        --is_training 1 \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done

