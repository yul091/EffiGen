
# method=NormKV # AdativeKV, ReasonKV, NormKV, fullkv
# max_capacity_prompts=128 # 128,2048 in paper
attn_implementation=flash_attention_2 # Support "flash_attention_2"
model_path=meta-llama/Meta-Llama-3-8B-Instruct
head_choice=('reason')
beta=1.005
temp=1
max_output_length=1


for max_capacity_prompts in 128 256 512 1024; do
    for model_path in meta-llama/Meta-Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2; do
        for method in fullkv NormKV ReasonKV AdativeKV; do
            save_dir="./results_profile/${method}/results_long_bench_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}"
            python run_latency.py \
                --method ${method} \
                --model_path ${model_path} \
                --max_capacity_prompts ${max_capacity_prompts} \
                --head_choice ${head_choice} \
                --beta ${beta} \
                --temp ${temp} \
                --attn_implementation ${attn_implementation} \
                --save_dir ${save_dir} \
                --use_cache True

            python run_memory.py \
                --method ${method} \
                --model_path ${model_path} \
                --max_capacity_prompts ${max_capacity_prompts} \
                --head_choice ${head_choice} \
                --beta ${beta} \
                --temp ${temp} \
                --attn_implementation ${attn_implementation} \
                --save_dir ${save_dir} \
                --length ${max_output_length} \
                --use_cache True
        done
    done
done
