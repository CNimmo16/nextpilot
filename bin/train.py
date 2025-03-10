import torch
from dataset import CodeDataset
from model import SimpleDecoder, tokenizer
import os
import shutil
from training import train_model
import util

# Config
DATA_DIR = "data/nextjs_repos"
CHECKPOINT_DIR = "data/checkpoints"
WEIGHTS_DIR = "data/weights"
BATCH_SIZE = 32
SEQ_LENGTH = 512
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
DISTILL_LOSS_TEMPERATURE = 0.7
DISTILL_LOSS_ALPHA = 0.7  # Weight between teacher and ground truth loss

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

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

if os.path.exists(CHECKPOINT_DIR):
    shutil.rmtree(CHECKPOINT_DIR)

os.makedirs(CHECKPOINT_DIR)

def on_epoch_done(epoch, model):
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
    }, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))

# Create a lower triangular matrix of ones
student_mask = torch.tril(torch.ones((SEQ_LENGTH - 1, SEQ_LENGTH - 1), device=DEVICE))

# Convert the mask to a suitable format for attention (0 for allowed, -inf for masked)
student_mask = student_mask.masked_fill(student_mask == 0, float('-inf'))

# Train
final_model = train_model(model, student_mask, train_loader, val_loader, optimizer, scheduler, DEVICE, epochs=EPOCHS, distill_loss_alpha=DISTILL_LOSS_ALPHA, distill_loss_temp=DISTILL_LOSS_TEMPERATURE, on_epoch_done=on_epoch_done)

id = util.generate_friendly_model_id(tokenizer)

if not os.path.exists(WEIGHTS_DIR):
    os.makedirs(WEIGHTS_DIR)

save_path = f"{WEIGHTS_DIR}/{id}.pth"

torch.save({
    'model_state_dict': model.state_dict(),
    'tokenizer': tokenizer,
}, save_path)

print(f"> Model saved to {save_path}")
