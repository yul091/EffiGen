######################################## For Longbench
max_capacity_prompts=128
export CUDA_LAUNCH_BLOCKING=1

devices=(0, 1, 2, 3, 4, 5, 6, 7,)
head_choices=('reason') # copy, reason
betas=(1.005 1.01 1.1 1.2 1.5 2 5 10)
counter=0
mkdir -p longbench_logs

for((j=0;j<1;j++));do
    head_choice=${head_choices[i]}
    beta=${betas[j]}
    temp=1
    nohup bash head_base.sh \
        $devices \
        fullkv \
        ${max_capacity_prompts} \
        flash_attention_2 \
        mistralai/Mixtral-8x7B-v0.1 \
        $head_choice \
        $beta \
        $temp > ./longbench_logs/mixtral_ReasonKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
    ((counter+=1))
done
# for((i=0;i<1;i++));do 
#     for((j=0;j<1;j++));do
#         device=${devices[counter]}
#         head_choice=${head_choices[i]}
#         beta=${betas[j]}
#         temp=1
#         bash head_base.sh \
#             $device \
#             ReasonKV \
#             ${max_capacity_prompts} \
#             flash_attention_2 \
#             mistralai/Mixtral-8x7B-Instruct-v0.1 \
#             $head_choice \
#             $beta \
#             $temp
#         ((counter+=1))
#     done
# done


# ###################### for babi-reason
# max_capacity_prompts=128 # 64, 128, 256, 512, 1024
# devices=(0 1 2 3 4 5 6 7)
# head_choices=('reason') # copy, reason
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
#             mistralai/Mistral-7B-Instruct-v0.2 \
#             $head_choice \
#             $beta \
#             $temp > ./reason_logs/mistral_ReasonKV_${head_choice}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
#         ((counter+=1))

#     done

# done
