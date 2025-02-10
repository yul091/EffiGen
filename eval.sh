
method=fullkv  # AdativeKV, ReasonKV, NormKV, fullkv
max_capacity_prompts=128  # 128,2048 in paper
model_name=mixtral-8x7b-v0.1  # meta-llama-3-8b-instruct, mistral-7b-instruct-v0.2, mixtral-8x7b-v0.1
beta=1.005
temp=1
# head_choices=('reason') # copy, reason

python eval.py \
    --results_dir ./results/${method}/results_long_bench_reason_base${max_capacity_prompts}_beta${beta}_temp${temp} \
    --model $model_name \
    --method $method \
    --capacity $max_capacity_prompts