"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack

# GPT-4
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider OpenAI\
    --model_name gpt-4-1106-preview
    --api_key $OPENAI_API_KEY
) 2>&1  | tee logs/eval_gpt_4_128k.log

# LLaMA 2 32K. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../Llama-2-7B-32K-Instruct
) 2>&1  | tee logs/eval_llama2_32k_instruct.log

# LongChat. Remember to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path /ML-A800/models/longchat-7b-v1.5-32k
) 2>&1  | tee logs/eval_longchat.log

# Our llama-2-7b-80k, requires 4*80G A100
# require you to download the model first
(
python -u needle_in_haystack.py --s_len 0 --e_len 128000\
    --model_provider LLaMA\
    --model_path ../../../llama-2-7b-80k
) 2>&1  | tee logs/eval_llama-2-7b-80k.log
"""

#import tiktoken
import os 
import glob
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import sys
sys.path.append("./faiss_attn/")
from source.modeling_llama import LlamaForCausalLM
from source.modeling_mixtral import MixtralForCausalLM
from source.modeling_mistral import MistralForCausalLM
import numpy as np
import argparse
from rouge_score import rouge_scorer
from datetime import datetime, timezone
from collections import defaultdict
import time
import torch



def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(seq_len=model_max_train_len, device=l.self_attn.rotary_emb.inv_freq.device, dtype=torch.float32)
    return
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """
    def __init__(self,
                needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
                haystack_dir="./haystack_for_detect",
                retrieval_question="What is the best thing to do in San Francisco?",
                results_version = 1,
                context_lengths_min = 1000,
                context_lengths_max = 50000,
                context_lengths_num_intervals = 5,
                context_lengths = None,
                document_depth_percent_min = 0,
                document_depth_percent_max = 100,
                document_depth_percent_intervals = 10,
                document_depth_percents = None,
                document_depth_percent_interval_type = "linear",
                model_provider = "OpenAI",
                model_name='',
                model_name_suffix=None,
                num_concurrent_requests = 1,
                save_results = True,
                save_contexts = False,
                final_context_length_buffer = 200,
                seconds_to_sleep_between_completions = None,
                print_ongoing_status = True):
        """        
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_provider: The provider of the model. Must be either 'OpenAI' or 'Anthropic'. Default is 'OpenAI'.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param seconds_to_sleep_between_completions: The number of seconds to sleep between completions. Default is None.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError("Needle, haystack, and retrieval_question must be provided.")
        needles_and_stacks = [json.loads(l) for l in open(f"{haystack_dir}/needles.jsonl")]
        self.needle_list = [l["needle"] for l in needles_and_stacks]
        self.haystack_dir_list = [f"{haystack_dir}/part{i}" for i in range(1, 4)]
        self.retrieval_question_list = [l["question"] for l in needles_and_stacks]
        self.real_ansers_list = [l["real_needle"] for l in needles_and_stacks]
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []
        self.head_counter = defaultdict(list)
        if("/" in model_name):
            self.model_version = model_name.split("/")[-1]
        else: self.model_version = model_name
        if(model_name_suffix is not None): self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if context_lengths_min is None or context_lengths_max is None or context_lengths_num_intervals is None:
                raise ValueError("Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied.")
            else:
                self.context_lengths = np.round(np.linspace(context_lengths_min, context_lengths_max, num=context_lengths_num_intervals, endpoint=True)).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if document_depth_percent_min is None or document_depth_percent_max is None or document_depth_percent_intervals is None:
                raise ValueError("Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied.")
            else:
                if document_depth_percent_interval_type == 'linear':
                    self.document_depth_percents = np.round(np.linspace(document_depth_percent_min, document_depth_percent_max, num=document_depth_percent_intervals, endpoint=True)).astype(int)
                elif document_depth_percent_interval_type == 'sigmoid':
                    self.document_depth_percents = [self.logistic(x) for x in np.linspace(document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals)]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError("document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals")
        
        self.model_name = model_name

        if self.model_name == 'mistralai/Mistral-7B-Instruct-v0.2':
            self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False, revision='dca6e4b60aca009ed25ffa70c9bb65e46960a573')
        else:
            self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
        print("loading from %s" % model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.layer_num, self.head_num = config.num_hidden_layers, config.num_attention_heads
        print(f"layer number: {self.layer_num}, head number {self.head_num}")
        if "Qwen" in self.model_version:
            self.model_to_test = Qwen2ForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2"
                ).eval()
        elif "Mixtral" in self.model_version:
            self.model_to_test = MixtralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        elif "Mistral" in self.model_version:
            self.model_to_test = MistralForCausalLM.from_pretrained(
                    model_name,torch_dtype="auto",device_map='auto',use_flash_attention_2="flash_attention_2",trust_remote_code=True,
                ).eval()
        else:
            self.model_to_test = LlamaForCausalLM.from_pretrained(model_name,
                use_flash_attention_2="flash_attention_2", torch_dtype=torch.bfloat16,device_map='auto').eval()
            
        if 'llama-2-7b-80k' in self.model_version:
            scaling_factor = 10
            reset_rope(self.model_to_test, model_max_train_len=81920, scaling_factor=scaling_factor)
            
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            self.multi_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"])>1
        else:
            self.multi_gpus = True
            
        self.model_to_test_description = model_name
        self.evaluation_model = None
        self.debug='debug'
        model_name = model_name.split('/')[-1]

    def logistic(self, x, L=100, x0=50, k=.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)
    
    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self, args):
        # Run through each iteration of context_lengths and depths
        tasks = []
         
        for context_length in self.context_lengths:
            if context_length < args.s_len or context_length > args.e_len: continue
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(context_length, depth_percent)

    def retrieval_calculate(self, attention_maxtrix,retrieval_score, inp, step_token,topk=1):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)
                for v, i in zip(values, idx):
                    if  self.needle_start <= i < self.needle_end and inp.item()==self.prompt_ids[i].item():
                        retrieval_score[layer_idx][head_idx][0] += 1/(self.needle_end - self.needle_start)
                        retrieval_score[layer_idx][head_idx][1] += step_token
                        break
    def retrieval_head_accumulate(self, retrieval_score):
        for layer_idx in range(self.layer_num):
            for head_idx in range(self.head_num):
                self.head_counter[f"{layer_idx}-{head_idx}"].append(retrieval_score[layer_idx][head_idx][0])

    def decode(self, q_outputs, inp, decode_len, block_list=None):
        output, retrieval_score = [], [[[0, ''] for _ in range(self.head_num)] for _ in range(self.layer_num)]
        past_kv = q_outputs.past_key_values
        for step_i in range(decode_len):
            inp = inp.view(1, 1)
            outputs = self.model_to_test(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch" )
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = self.enc.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            self.retrieval_calculate(outputs.attentions, retrieval_score, inp, step_token)
            if step_token=='<0x0A>' or inp.item()==144: break
        return output, retrieval_score 

    def find_needle_idx(self, needle):
        needle_ids = self.enc(needle, add_special_tokens=False)["input_ids"]
        print( self.enc.decode(needle_ids, skip_special_tokens=False))
        span_len = len(needle_ids)
        for i in range(len(self.prompt_ids)):            
            token_span = self.prompt_ids[i : i + span_len]
            span_ids = set(token_span.tolist())
            overlap = float(len(span_ids.intersection(set(needle_ids)))) / len(set(needle_ids))
            if(overlap > 0.9):
                return i, i + span_len
        return -1, -1

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)
        question = f"Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
        '''
        if self.model_version=="Qwen1.5-14B-Chat":
            context = "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n" + context input_context = "f{context}\nquestion<|im_end|>\n<|im_start|>assistant\n
            question += '<|im_end|>\n<|im_start|>assistant\n'
            input_ids = self.enc(input_context , return_tensors="pt")['input_ids']
        '''
        # if self.model_version in ["Mistral-7B-Instruct-v0.2", "Qwen1.5-14B-Chat"]:
        if 'Qwen' in self.model_version:
            prompt = [
            {"role": "user", "content": f"<book>{context}</book>\nBased on the content of the book, Question: {self.retrieval_question}\nAnswer:"},
            ]
            input_context = self.enc.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )        
        else:
            input_context = context + question
        
        input_ids = self.enc(input_context , return_tensors="pt")['input_ids']
        
        # Prepare your message to send to the model you're going to evaluate
        test_start_time = time.time()
        self.prompt_ids = input_ids[0, :]
        if not self.multi_gpus:
            input_ids = input_ids.to(self.model_to_test.device)
        self.needle_start, self.needle_end = self.find_needle_idx(self.real_needle)
        with torch.no_grad():
            q_outputs = self.model_to_test(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)
            output, retrieval_score  = self.decode(q_outputs, input_ids[:,-1], 50)
            response = self.enc.decode(output,skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time
        
        score = scorer.score(self.real_needle, response)['rouge1'].recall*100
        ## if recall > 50, we determine this retrieval succeed and update the retrieval score
        if score > 50:
            self.retrieval_head_accumulate(retrieval_score)
            head_score = [(i[0], np.mean(i[1])) for i in self.head_counter.items()]
            head_score = sorted(head_score, key=lambda x:x[1], reverse=True)
            print([[i[0]] for i in head_score][:20])

        results = {
            'model' : self.model_to_test_description,
            'context_length' : int(context_length),
            'depth_percent' : float(depth_percent),
            'version' : self.results_version,
            'needle' : self.needle,
            'model_response' : response,
            'score' : score,
            'test_duration_seconds' : test_elapsed_time,
            'test_timestamp_utc' : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print (f"-- Test Summary -- ")
            print (f"Duration: {test_elapsed_time:.1f} seconds")
            print (f"Context: {context_length} tokens")
            print (f"Depth: {depth_percent}%")
            print (f"Score: {score}")
            print (f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent*100)}'

        if self.save_contexts:
            results['file_name'] : context_file_location

            # Save the context to file for retesting
            if not os.path.exists('contexts'):
                os.makedirs('contexts')

            if not os.path.exists(f'contexts/{self.model_version}'):
                os.makedirs(f'contexts/{self.model_version}')

            with open(f'contexts/{self.model_version}/{context_file_location}_context.txt', 'w') as f:
                f.write(context)
            
        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(f'results/graph/{self.model_version}'):
                os.makedirs(f'results/graph/{self.model_version}')
            
            # Save the result to file for retesting
            p = f'results/graph/{self.model_version}/{context_file_location}_results.json'
            print("Writing at %s" % p)
            with open(p, 'w') as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = 'results/' + self.model_version
        print("Searching existing results at %s" % results_dir)
        if not os.path.exists(results_dir):
            return False
        for filename in os.listdir(results_dir):
            if filename.endswith('.json'):
                with open(os.path.join(results_dir, filename), 'r') as f:
                    result = json.load(f)
                    context_length_met = result['context_length'] == context_length
                    depth_percent_met = result['depth_percent'] == depth_percent
                    version_met = result.get('version', 1) == self.results_version
                    model_met = result['model'] == self.model_name
                    # import ipdb; ipdb.set_trace()
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context
    
    def encode_text_to_tokens(self, text):
        if self.model_provider in ["OpenAI", "LLaMA", "LLAMA3", "Mistral", "GLM"]:
            return self.enc.encode(text)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(text).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
    
    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            # import ipdb; ipdb.set_trace()

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            if(self.model_provider in ["LLaMA", "LongLLaMA"]): period_tokens = [29889, 869]
            elif(self.model_provider == 'LLAMA3'): period_tokens = [13]
            elif(self.model_provider == "Mistral"): period_tokens = [842, 28723]
            elif(self.model_provider == "GLM"): period_tokens = [918, 30930]
            else: period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "LLAMA3", "Mistral","GLM"]:
            return len(self.enc.encode(context))
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            encoded = self.enc.encode(context)
            return len(self.enc.encode(context).ids)
        else:
            
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while len(context.split()) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, 'r') as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        if self.model_provider in ["OpenAI", "LLaMA", "LLAMA3",  "Mistral", "GLM"]:
            return self.enc.encode(context)
        elif self.model_provider == "Anthropic":
            # Assuming you have a different encoder for Anthropic
            return self.enc.encode(context).ids
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")
        
    def decode_tokens(self, tokens, context_length=None):
        if self.model_provider in ["OpenAI", "LLaMA", "LLAMA3", "Mistral", "GLM"]:
            return self.enc.decode(tokens[:context_length])
        elif self.model_provider == "Anthropic":
            # Assuming you have a different decoder for Anthropic
            return self.enc.decode(tokens[:context_length])
        else:
            raise ValueError("model_provider must be either 'OpenAI' or 'Anthropic'")

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context
    
    def get_results(self):
        return self.testing_results
    
    def print_start_test_summary(self):
        print ("\n")
        print ("Starting Needle In A Haystack Testing...")
        print (f"- Model: {self.model_name}")
        print (f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}")
        print (f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%")
        print (f"- Needle: {self.needle.strip()}")
        print ("\n\n")

    def start_test(self, args):
        for ni in range(len(self.needle_list)):
            self.needle = self.needle_list[ni]
            self.haystack_dir = self.haystack_dir_list[ni]
            self.real_needle  = self.real_ansers_list[ni]
            self.retrieval_question = self.retrieval_question_list[ni]
            if self.print_ongoing_status:
                self.print_start_test_summary()
            self.run_test(args)
        if os.path.exists(f"head_score/{self.model_version}_retrieval_heads.json"):
            with open(f"./head_score/{self.model_version}_retrieval_heads.json", "r") as file:
                head_counter = json.loads(file.readline())
            for k,v in head_counter.items():
                self.head_counter[k] += v
        with open(f"head_score/{self.model_version}_retrieval_heads.json", 'w') as f:
            json.dump(self.head_counter, f)


if __name__ == "__main__":
    # Tons of defaults set, check out the LLMNeedleHaystackTester's init for more info
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--s_len', metavar='N', type=int, help='a number')
    parser.add_argument('-e', '--e_len', metavar='N', type=int, help='a number')
    parser.add_argument('--model_path', type=str, default=None, help='path to model')
    parser.add_argument('--model_name', type=str, default=None, help='name of model')
    parser.add_argument('--model_name_suffix', type=str, default=None, help='name of model')
    parser.add_argument('--model_provider', type=str, default="LLaMA", help='which model to use')
    args = parser.parse_args()
   
    model_name = args.model_path


    ht = LLMNeedleHaystackTester(model_name=model_name, 
                                 model_name_suffix=args.model_name_suffix,
                                 model_provider=args.model_provider,
                                 save_contexts=False,
                                 save_results=False,
                                 context_lengths_min=args.s_len,
                                 context_lengths_max=args.e_len,
                                 )

    ht.start_test(args)