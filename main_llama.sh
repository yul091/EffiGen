NUM_SAMPLES=334
MAX_SERVING_BATCH_SIZE=6
MAX_TRAINING_BATCH_SIZE=4
MEMORY_THRESHOLD=0.9
BASE_DIR=prof_main
EXPERIMENTS=1  # each experiment run 1 times
export CUDA_LAUNCH_BLOCKING=1
MODEL_NAME="Llama-2-7b-chat-hf" # "Llama-2-7b-chat-hf" "Llama-2-13b-chat-hf" "Llama-2-70b-chat-hf"
# Ensure the folder for profiling output exists
DEVICE=0
PROFILING_DIR="nvidia-profiling"
mkdir -p ${PROFILING_DIR}

for MODEL_NAME in "Llama-2-7b-chat-hf"; do
    for RATE_LAMBDA in -1; do
        # if lambda is -1, then we use alternate for output_dir
        if [ $RATE_LAMBDA -eq -1 ]; then
            OUTPUT_DIR=${BASE_DIR}/lambda_alternate/${MODEL_NAME}
        else
            OUTPUT_DIR=${BASE_DIR}/lambda_${RATE_LAMBDA}/${MODEL_NAME}
        fi
        # for RETRAIN_RATE in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        for RETRAIN_RATE in 0.0; do
            # nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw --format=csv,nounits -l 1 -f ${PROFILING_DIR}/${TASK_ASSIGNMENT}_${MODEL_NAME}_${NUM_NODES}-node_${RATE_LAMBDA}-rps_${RETRAIN_RATE}-rate.csv &
            # NVIDIA_SMI_PID=$!
            python mix_pipeline_sequential.py \
                --model_name_or_path "meta-llama/${MODEL_NAME}" \
                --dataset_name_or_path "data/Anthropic(old)" \
                --model_name $MODEL_NAME \
                --n_samples $NUM_SAMPLES \
                --rate_lambda $RATE_LAMBDA \
                --retraining_rate $RETRAIN_RATE \
                --output_dir $OUTPUT_DIR \
                --serving_batch_size $MAX_SERVING_BATCH_SIZE \
                --training_batch_size $MAX_TRAINING_BATCH_SIZE \
                --experiments $EXPERIMENTS \
                --memory_threshold $MEMORY_THRESHOLD \
                --device $DEVICE \
                --run_mode 'online'
            
            # kill $NVIDIA_SMI_PID
            # python plot.py \
            #     --node $NUM_NODES \
            #     --model_name $MODEL_NAME \
            #     --setting $SETTING \
            #     --workload $WORKLOAD \
            #     --task_assignment $TASK_ASSIGNMENT \
            #     --retraining_rate $RETRAIN_RATE \
            #     --alpha $ALPHA \
            #     --beta $BETA \
            #     --epsilon $EPSILON \
            #     --output_dir $OUTPUT_DIR

        done
    done
done

