import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
import os
import time
import torch
import cml.metrics_v1 as metrics
import cml.models_v1 as models

# Environment setup
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
hf_access_token = os.environ.get('HF_ACCESS_TOKEN')

# Quantization
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True, 
    load_in_4bit=True
)

# Create a model object with above parameters
model_name = "CohereForAI/c4ai-command-r-v01-4bit"

# Initialize model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        offload_folder="offload",
        offload_state_dict=True,
        token=hf_access_token
    )

# Infer device map
device_map = infer_auto_device_map(model, max_memory={"cpu": "48GB", "cuda:0": "16GB"})

# Dispatch model according to the inferred device map
model = dispatch_model(model, device_map=device_map)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Define tokenizer parameters
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_access_token)
tokenizer.pad_token = tokenizer.eos_token

# Args helper
def opt_args_value(args, arg_name, default):
    """
    Helper function to interact with LLMs parameters for each call to the model.
    Returns value provided in args[arg_name] or the default value provided.
    """
    if arg_name in args.keys():
        return args[arg_name]
    else:
        return default

# Generate response function
def generate(prompt, max_new_tokens=50, temperature=0, repetition_penalty=1.0, num_beams=1, top_p=1.0, top_k=0):
    """
    Make a request to the LLM, with given parameters (or using default values).

    max_new_tokens     - at how many words will the generated response be capped?
    temperature        - a.k.a. "response creativity". Controls randomness of the generated response (0 = least random, 1 = more random). 
    repetition_penalty - penalizes the next token if it has already been used in the response (1 = no penalty)
    num_beams          - controls the number of token sequences generate (1 = only one sequence generated)
    top_p              - cumulative probability to determine how many tokens to keep (i.e. enough tokens will be considered, so their combined probability reaches top_p)
    top_k              - number of highest-probability tokens to keep (i.e. only top_k "best" tokens will be considered for response)
    """
    batch = tokenizer(prompt, return_tensors='pt').to("cuda")
  
    with torch.cuda.amp.autocast():
        output_tokens = model.generate(**batch,
                                       max_new_tokens=max_new_tokens,
                                       repetition_penalty=repetition_penalty,
                                       temperature=temperature,
                                       num_beams=num_beams,
                                       top_p=top_p,
                                       top_k=top_k)
  
    output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
  
    # Log the response along with parameters
    print("Prompt: %s" % (prompt))
    print("max_new_tokens: %s; temperature: %s; repetition_penalty: %s; num_beams: %s; top_p: %s; top_k: %s" % (max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k))
    print("Full Response: %s" % (output))
  
    return output

@models.cml_model(metrics=True)
def api_wrapper(args):
    """
    Process an incoming API request and return a JSON output.
    """
    start = time.time()
  
    # Pick up args from model api
    prompt = args["prompt"]
  
    # Pick up or set defaults for inference options
    temperature = float(opt_args_value(args, "temperature", 0))
    max_new_tokens = float(opt_args_value(args, "max_new_tokens", 50))
    top_p = float(opt_args_value(args, "top_p", 1.0))
    top_k = int(opt_args_value(args, "top_k", 0))
    repetition_penalty = float(opt_args_value(args, "repetition_penalty", 1.0))
    num_beams = int(opt_args_value(args, "num_beams", 1))
  
    # Generate response from the LLM
    response = generate(prompt, max_new_tokens, temperature, repetition_penalty, num_beams, top_p, top_k)
  
    # Calculate elapsed time
    response_time = time.time() - start
  
    # Track model outputs over time
    metrics.track_metric("prompt", prompt)
    metrics.track_metric("response", response)
    metrics.track_metric("response_time_s", response_time)

    return {"response": response, "response_time_s": round(response_time, 1)}
