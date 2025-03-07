import os
import torch
from model import SimpleDecoder, get_llama
from transformers import CodeLlamaTokenizer
from tokenizer import tokenizer
import tqdm
import random

torch.manual_seed(16)
random.seed(16)

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

            code = f"<PRE>{code}<SUF>"
            
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

TEMPERATURE = 0.7
ALPHA = 0.7  # Weight between teacher and ground truth loss
def distill_loss(student_logits, teacher_logits, labels):
    # Soften teacher logits with temperature
    soft_teacher = torch.nn.functional.softmax(teacher_logits / TEMPERATURE, dim=-1)
    
    # Calculate distillation loss (KL divergence)
    loss_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / TEMPERATURE, dim=-1),
        soft_teacher,
        reduction="batchmean"
    ) * (TEMPERATURE ** 2)
    
    # Calculate standard cross-entropy loss
    loss_ce = torch.nn.functional.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=tokenizer.pad_token_id
    )
    
    return ALPHA * loss_kl + (1 - ALPHA) * loss_ce

def train_model(student: SimpleDecoder, train_loader, val_loader, optimizer, device, epochs, on_epoch_done=None):
    teacher = get_llama()
    teacher.eval()

    for epoch in range(epochs):
        print(f"==== Epoch {epoch+1} ====")
        total_train_loss = 0

        student.train()

        pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, batch in pbar:
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=inputs,
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits
            
            # Get student predictions
            student_logits = student(inputs)
            
            # Compute distillation loss
            loss = distill_loss(student_logits, teacher_logits, inputs)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

            avg_loss = total_train_loss / (batch_idx + 1)

            pbar.set_description(f"Training... (avg. distillation loss {avg_loss:.3f})")

        student.eval()

        total_val_loss = 0
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating..."):
            inputs = batch["input_ids"].to(device)
            
            outputs = student(inputs)
            
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), inputs.view(-1), ignore_index=tokenizer.pad_token_id)
            
            total_val_loss += loss.item()

        print(f"Completed Epoch {epoch+1}: Average Train Loss: {total_train_loss/len(train_loader):.4f}, Average Val Loss: {total_val_loss/len(val_loader):.4f}")
            
        if on_epoch_done is not None:
            on_epoch_done(epoch, student)

# 4. Main execution
if __name__ == "__main__":
    # Config
    DATA_DIR = "data/nextjs_repos"
    MODEL_SAVE_DIR = "data/weights"
    BATCH_SIZE = 1
    SEQ_LENGTH = 512
    EPOCHS = 100
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    LEARNING_RATE = 0.0003

    # Initialize components
    dataset = CodeDataset(DATA_DIR, tokenizer, max_length=SEQ_LENGTH)

    # Get train & val datasets
    val_split = 0.1
    max_val_count = 5000
    data_count = len(dataset)
    val_count = round(max(min(data_count * val_split, max_val_count), 1))
    train = torch.utils.data.Subset(dataset, range(data_count - val_count))
    val = torch.utils.data.Subset(dataset, range(data_count - val_count, data_count))

    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = SimpleDecoder(vocab_size=len(tokenizer), max_seq_len=SEQ_LENGTH).to(DEVICE)
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    def on_epoch_done(epoch, model):
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer': tokenizer,
        }, os.path.join(MODEL_SAVE_DIR, f"nextjs_decoder_epoch_{epoch}.pth"))

    # Train
    train_model(model, train_loader, val_loader, optimizer, DEVICE, epochs=EPOCHS, on_epoch_done=on_epoch_done)
