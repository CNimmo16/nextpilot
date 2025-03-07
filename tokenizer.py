from transformers import CodeLlamaTokenizer

tokenizer: CodeLlamaTokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
