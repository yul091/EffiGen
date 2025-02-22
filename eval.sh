
method=AdativeKV  # AdativeKV, ReasonKV, NormKV, fullkv, SnapKV, PyramidKV
max_capacity_prompts=128  # 128,2048 in paper
model_name=meta-llama-3-8b-instruct  # meta-llama-3-8b-instruct, mistral-7b-instruct-v0.2, mixtral-8x7b-instruct-v0.1
beta=1.005
temp=1
# head_choices=('reason') # copy, reason

for max_capacity_prompts in 128 256 512 1024; do
# for max_capacity_prompts in 1024; do
    for method in SnapKV; do
        python eval.py \
            --results_dir ./results/${method}/results_long_bench_reason_base${max_capacity_prompts}_beta${beta}_temp${temp} \
            --model $model_name \
            --method $method \
            --capacity $max_capacity_prompts
    done
done

# method=fullkv 
# model_name=mistral-7b-instruct-v0.2
# python eval.py \
#     --results_dir ./results/${method}/results_long_bench_reason_base${max_capacity_prompts}_beta${beta}_temp${temp} \
#     --model $model_name \
#     --method $method \
#     --capacity $max_capacity_prompts
