import os
import torch
import torch.nn as nn
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from model import SimpleDecoder
from transformers import LlamaForCausalLM, CodeLlamaTokenizer

# 2. Dataset class
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.tokenizer: CodeLlamaTokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'r', encoding='utf-8') as f:
            code = f.read()
            # Remove comment headers
            code = '\n'.join([line for line in code.split('\n') if not line.startswith('//')])
            
            # Tokenize and add special tokens
            tokens = self.tokenizer.encode(code)
            tokens = self.tokenizer.convert_tokens_to_ids(["<s>"]) + tokens + self.tokenizer.convert_tokens_to_ids(["</s>"])
            
            # Split into chunks of max_length
            chunks = []
            for i in range(0, len(tokens), self.max_length):
                chunk = tokens[i:i+self.max_length]
                if len(chunk) < self.max_length:
                    chunk += self.tokenizer.convert_tokens_to_ids(["<pad>"]) * (self.max_length - len(chunk))
                chunks.append(chunk)
            return torch.tensor(chunks, dtype=torch.long)

# 3. Training loop
def train(model, dataloader, optimizer, device, epochs=5):

    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Create shifted inputs/targets
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Reshape for loss calculation
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Loss: {total_loss/len(dataloader):.4f}")

# 4. Main execution
if __name__ == "__main__":
    # Config
    DATA_DIR = "data/nextjs_repos"
    MODEL_SAVE_DIR = "data/weights"
    BATCH_SIZE = 16
    SEQ_LENGTH = 512
    EPOCHS = 30
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 3e-4

    # Initialize components
    tokenizer: CodeLlamaTokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")
    dataset = CodeDataset(DATA_DIR, tokenizer, max_length=SEQ_LENGTH)

    def collate(batch):
        list_of_all = torch.tensor([], dtype=torch.long)
        for item in batch:
            list_of_all = torch.cat((list_of_all, item), 0)
        return list_of_all
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    # Initialize model
    model = SimpleDecoder(vocab_size=tokenizer.vocab_size, max_seq_len=SEQ_LENGTH).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Train
    train(model, dataloader, optimizer, DEVICE, epochs=EPOCHS)

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
    }, os.path.join(MODEL_SAVE_DIR, 'nextjs_decoder.pth'))
