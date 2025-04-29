from model_inference import InferenceRun

from peft import PeftModel
from transformers import AutoTokenizer
import json
from tqdm import tqdm

class FineTunedInference(InferenceRun):

    def __init__(self, run_name: str, run_config: dict):
        """Constructor for FineTunedInference class. This class assists with loading a fine tuned PEFT model
        and running inference tasks. Inherits Huggingface base model functionality from InferenceRun class.
        run_name: str - name of the inference run
        run_config: dict - dictionary of items needed to configure the inference run"""
        super().__init__(run_name, run_config)
        self.fine_tuned_folder = run_config['fine_tuned_dir']
        self.load_finetuned_model()

    
    def load_finetuned_model(self):
        """Function to load a model from Bloomberg Lab S3 storage and apply the fine-tuned layers to a base model"""
        # load the base model first
        base_model = super().load_model_from_storage(self.model_s3_loc)
        # clear the folder this has been downloaded
        self.helper.clear_folder(self.model_s3_loc)
        # create the fine_tuned_model
        self.fine_tuned_model = PeftModel.from_pretrained(base_model, self.fine_tuned_folder)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_hf_id)

    
    def reformat_prompts(self, prompt_set: str, message: str):
        """Function to reformat the prompts to match the style used to finetune the model"""
        new_prompts = []
        for prompt in prompt_set:
            prompt_dict = {
                'security': prompt['security'],
                'date': prompt['date'],
                'prompt': prompt['prompt'][0]['content'] + '\n' + prompt['prompt'][1]['content'] + message
            }
            new_prompts.append(prompt_dict)
        return new_prompts
            
        
    def run_finetuned_backtest(self, prompts: list):
        """Function to run inference tasks over a fine-tuned model
        prompts: list - list of dictionary items for the inference run."""
        outputs = []
        #count = 0
        progress = tqdm(total=len(prompts), position=0, leave=True)

        # Loop through each of the prompts
        for prompt in prompts:
            # tokenize the prompt
            model_inputs = self.tokenizer([prompt['prompt']], return_tensors='pt').to("cuda")
            # generate output from fine-tuned llm
            generated_ids = self.fine_tuned_model.generate(**model_inputs, 
                                           pad_token_id=self.tokenizer.eos_token_id, 
                                           max_new_tokens=50,
                                          temperature=0.001)
            parsed_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            resp = {
                'date': prompt['date'],
                'security': prompt['security'],
                'response': self.tokenizer.batch_decode(parsed_ids, skip_special_tokens=True)[0]
            }
            outputs.append(resp)
            progress.update()
        return outputs