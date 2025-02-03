# parser.add_argument('--results_dir', type=str, default=None)
# parser.add_argument('--model', type=str, default='meta-llama-3-8b-instruct')
# parser.add_argument('--capacity', type=int, default=128)
# parser.add_argument('--longbench_e', action='store_true', help="Evaluate on LongBench-E")

# mistralai/Mistral-7B-Instruct-v0.2
python eval.py \
    --results_dir results-norm-value/results_long_bench_reason_base128_beta1.005_temp1 \
    --model meta-llama-3-8b-instruct \
    --capacity 128