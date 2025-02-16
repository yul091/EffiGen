export CUDA_LAUNCH_BLOCKING=1
# method=NormKV # AdativeKV, ReasonKV, NormKV, NormKV_indexmlp, fullkv, SnapKV, PyramidKV
# max_capacity_prompts=128 # 128,2048 in paper
attn_implementation=flash_attention_2 # Support "flash_attention_2", "eager"
model_path=mistralai/Mixtral-8x7B-Instruct-v0.1 # meta-llama/Meta-Llama-3-8B-Instruct, allenai/OLMoE-1B-7B-0924, mistralai/Mistral-7B-Instruct-v0.2, mistralai/Mixtral-8x7B-Instruct-v0.1
head_choice=('reason')
beta=1.005
temp=1
max_output_length=1
# device=5
# --device ${device} \

for max_capacity_prompts in 128; do
    # for model_path in mistralai/Mixtral-8x7B-Instruct-v0.1; do
    # for method in PyramidKV; do
    for method in fullkv; do
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
            --prune_mlp \
            --sparsity 0.5 \
            --use_cache True

        # python run_memory.py \
        #     --method ${method} \
        #     --model_path ${model_path} \
        #     --max_capacity_prompts ${max_capacity_prompts} \
        #     --head_choice ${head_choice} \
        #     --beta ${beta} \
        #     --temp ${temp} \
        #     --attn_implementation ${attn_implementation} \
        #     --save_dir ${save_dir} \
        #     --length ${max_output_length} \
        #     --use_cache True
    done
    # done
done
