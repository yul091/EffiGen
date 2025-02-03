######################################## For Longbench
max_capacity_prompts=128
export CUDA_LAUNCH_BLOCKING=1

devices=(0 1 2 3 4 5 6 7 8)
head_choices=('reason') #  copy, reason
betas=(1.005 1.01 1.1 1.2 1.5 2 5 10)
counter=0
# Create longbench_logs directory if it does not exist
mkdir -p longbench_logs
# for((i=0;i<1;i++));do 
#     for((j=0;j<1;j++));do
#         device=${devices[counter]}
#         head_choice=${head_choices[i]}
#         beta=${betas[j]}
#         temp=1
#         nohup bash head_base.sh \
#             $device \
#             ReasonKV \
#             ${max_capacity_prompts} \
#             flash_attention_2 \
#             meta-llama/Meta-Llama-3-8B-Instruct \
#             $head_choice \
#             $beta \
#             $temp > ./longbench_logs/llama3_ReasonKV-consistent_${head_choice}_base${max_capacity_prompts}_temp${temp}.txt 2>&1 &
#         ((counter+=1))
#     done
# done
for ((i=0; i<1; i++)); do 
    for ((j=0; j<1; j++)); do
        device=${devices[counter]}
        head_choice=${head_choices[i]}
        beta=${betas[j]}
        temp=1
        bash head_base.sh \
            $device \
            ReasonKV \
            ${max_capacity_prompts} \
            flash_attention_2 \
            meta-llama/Meta-Llama-3-8B-Instruct \
            $head_choice \
            $beta \
            $temp
        ((counter+=1))
    done
done



# #####################  for babi-reason
# max_capacity_prompts=128 
# head_choices=('reason') # copy, reason
# # betas=(1.02 1.005 1.1 1.2 1.5 2 5 10) # 64 
# betas=(1.005 1.01 1.1 1.2 1.5 2 5 10) # 128 256 512 

# counter=0
# for((i=0;i<1;i++));do 
#     for((j=0;j<8;j++));do
#         device=${devices[counter]}
#         head_choice=${head_choices[i]}
#         beta=${betas[j]}
#         temp=1
#         nohup bash head_base_babi.sh \
#             $device \
#             ReasonKV \
#             ${max_capacity_prompts} \
#             flash_attention_2 \
#             meta-llama/Meta-Llama-3-8B-Instruct \
#             $head_choice \
#             $beta \
#             $temp > ./reason_logs/llama3_ReasonKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
#         ((counter+=1))

#     done

# done
