for data_file in ECT2.csv; do
  for target in mps; do
    for len in 576; do
      task_name="${data_file%%.*}_${len}_mul"
      echo "Running diff with task_name=$task_name" | tee -a exp_result.txt
      python -u run.py \
        --dir_path "./data/" \
        --data_path "$data_file" \
        --seq_len "$len" \
        --model "HRS" \
        --pred_len "$len" \
        --hidden_dim 64 \
        --patch_size 16 20 \
        --stride 8 10 \
        --dimension_mlp_dim 64 \
        --token_mlp_dim 64 \
        --h 200 \
        --freq "t" \
        --learning_rate 0.0001 \
        --lradj "type1" \
        --target "$target" \
        --patience 10 \
        --is_training 1 \
        --task_id "$task_name" 2>&1 | tee -a exp_result.txt
      echo "Completed task: $task_name" | tee -a exp_result.txt
      echo "---------------------------------" | tee -a exp_result.txt
    done
  done
done

