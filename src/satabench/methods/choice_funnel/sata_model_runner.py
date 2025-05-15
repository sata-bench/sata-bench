from transformers import AutoModelForCausalLM, AutoTokenizer
import string
import torch.nn.functional as F
import torch
import numpy as np
import os


class SATAModelRunner:
    def __init__(self, model_id, option_prefix="", encode_option: bool=False):
        """
        Args:
            encode_option: for models like Phi-3 where top labels generated are bytes.
        """
        self.option_prefix = option_prefix
        self.encode_option = encode_option
        self.model_id = model_id

        hf_token = os.environ.get("HF_TOKEN")
        token_kwargs = {"token": hf_token} if hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            **token_kwargs
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_id,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="cuda",
                                                          trust_remote_code=True,
                                                          **token_kwargs,
        )

        # turn on flash attn helps given we only do prefill to get first token
        self.model.use_flash_attn = False

        # initialize the token ids which model will be generating in MCQ setting
        self.atoz = list(string.ascii_uppercase)
        self.option_prefix = option_prefix
        if encode_option:
            options = [option_prefix + i.encode() for i in self.atoz]
        else:
            options = [option_prefix + i for i in self.atoz]
        self.option_ids = [self.tokenizer.convert_tokens_to_ids(o) for o in options]

    def generate(self, model_input: str, num_options: int, **hf_generate_args):
        if not hf_generate_args:
            hf_generate_args = {
                'max_new_tokens': 1,
                'do_sample': False,
                'output_scores': True,
                'return_dict_in_generate': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }

        model_input = self.tokenizer(model_input, return_tensors="pt").to("cuda")
        output = self.model.generate(**model_input, **hf_generate_args)

        # Get logits for the next token prediction
        option_ids = self.option_ids[:num_options]
        options = self.atoz[:num_options]

        # extract option ids log probs
        option_logits = output.scores[0][:, option_ids].to("cpu")
        option_probs = F.softmax(option_logits, dim=-1)
        option_prob_dict = {option: prob.item() for option, prob in zip(options, option_probs[0])}
        sorted_option_prob_dict = dict(sorted(option_prob_dict.items(), key=lambda item: item[1], reverse=True))

        prediction = ''.join([option for option, prob in sorted_option_prob_dict.items()])
        return sorted_option_prob_dict, prediction

    def generate_batch(self, model_inputs: list, num_options: list, **hf_generate_args):
        if not hf_generate_args:
            hf_generate_args = {
                'max_new_tokens': 1,
                'do_sample': False,
                'output_scores': True,
                'return_dict_in_generate': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }
    
        # Ensure num_options matches the number of inputs
        if len(model_inputs) != len(num_options):
            raise ValueError("The length of `model_inputs` must match the length of `num_options`.")
    
        # Tokenize inputs as a batch
        model_inputs = self.tokenizer(model_inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
    
        # Generate outputs for the batch
        output = self.model.generate(**model_inputs, **hf_generate_args)
    
        results = []
        # Process each batch item
        for batch_idx, num_option in enumerate(num_options):
            # Prepare option IDs and labels for the current input
            option_ids = self.option_ids[:num_option]
            options = self.atoz[:num_option]
    
            # Extract logits for the current batch item
            option_logits = output.scores[0][batch_idx][option_ids].to("cpu")
            option_probs = F.softmax(option_logits, dim=-1)
    
            # Create probability dictionary and sort it
            option_prob_dict = {option: prob.item() for option, prob in zip(options, option_probs)}
            sorted_option_prob_dict = dict(sorted(option_prob_dict.items(), key=lambda item: item[1], reverse=True))
    
            # Generate the prediction
            prediction = ''.join([option for option, prob in sorted_option_prob_dict.items()])
            results.append((sorted_option_prob_dict, prediction))
    
        return results

    def generate_yesno(self, model_input: str, **hf_generate_args):
        """
        Model generation method designed for yes/no question type
        We extracts the logits for yes and no tokens then return
        True if yes logits is higher than no, else return False
        """
        if not hf_generate_args:
            hf_generate_args = {
                'max_new_tokens': 1,
                'do_sample': False,
                'output_scores': True,
                'return_dict_in_generate': True,
                'pad_token_id': self.tokenizer.eos_token_id
            }

        model_input = self.tokenizer(model_input, return_tensors="pt").to("cuda")
        output = self.model.generate(**model_input, **hf_generate_args)

        # Postprocessing
        if self.encode_option:
            options = [self.option_prefix+"Yes".encode(), self.option_prefix+"No".encode()]
        else:
            options = [self.option_prefix+"Yes", self.option_prefix+"No"]
        option_ids = [self.tokenizer.convert_tokens_to_ids(o) for o in options]

        # option_probs = F.softmax(output.scores[0], dim=-1)[:, option_ids]
        option_logits = output.scores[0][:, option_ids].to("cpu")
        option_probs = F.softmax(option_logits, dim=-1)
        return np.argmax(option_logits) == 0, option_probs[0][np.argmax(option_logits)]

    def hf_generate(self, model_input: str, **hf_generate_args):
        model_input = self.tokenizer(model_input, return_tensors="pt").to("cuda")
        output = self.model.generate(**model_input, **hf_generate_args)
        return output
