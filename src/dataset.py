import torch
from transformers import CodeLlamaTokenizer
import os

class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, _tokenizer: CodeLlamaTokenizer, max_length=512):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.tokenizer = _tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            code = f.read()
            # Remove comment headers
            code = '\n'.join([line for line in code.split('\n') if not line.startswith('//')])
            
            inputs = self.tokenizer(
                code,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length"
            )

            return {
                "input_ids": inputs['input_ids'].squeeze(),
                "attention_mask": inputs['attention_mask'].squeeze(), 
            }
