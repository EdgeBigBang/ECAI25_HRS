for data_file in ECT1.csv; do
  for target in mps; do
    for len in 864; do
      task_name="${data_file%%.*}_${len}"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --seq_len "$len" \
        --model "HRS" \
        --pred_len "$len" \
        --h 140 \
        --lw 4 \
        --expand 1 \
        --channel 1 \
        --hidden_dim 8 \
        --dimension_mlp_dim 16 \
        --patch_size 4 4 \
        --stride 2 2 \
        --token_mlp_dim 32 \
        --n_blocks 11 \
        --dropout 0 \
        --learning_rate 0.003 \
        --freq "t" \
        --lradj "type1" \
        --patience 10 \
        --target "$target" \
        --is_training 1 \
        --inverse 1 \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done

