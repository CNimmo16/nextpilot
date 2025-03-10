import torch
from src.model import SimpleDecoder, get_llama
from src.tokenizer import tokenizer
import tqdm
import src.util
import wandb

torch.manual_seed(16)

def distill_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # Soften teacher logits with temperature
    soft_teacher = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    
    # Calculate distillation loss (KL divergence)
    loss_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
        soft_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)
    
    # Calculate standard cross-entropy loss
    loss_ce = torch.nn.functional.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), labels.reshape(-1), ignore_index=tokenizer.pad_token_id)
    
    return alpha * loss_kl + (1 - alpha) * loss_ce

def train_model(
    student: SimpleDecoder,
    student_mask,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ChainedScheduler,
    device,
    epochs,
    distill_loss_temp: float,
    distill_loss_alpha: float,
    on_epoch_done=None
):
    teacher = get_llama(device)
    teacher.eval()

    wandb.init(project='distillation')
    
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
                    input_ids=inputs[:, :-1],
                    attention_mask=attention_mask
                )
                teacher_logits = teacher_outputs.logits

            optimizer.zero_grad()

            target_inputs = inputs[:, 1:]  # Shift right to create targets
            student_logits = student(inputs[:, :-1], mask=student_mask).logits
            
            # Compute distillation loss
            loss = distill_loss(student_logits, teacher_logits, target_inputs, distill_loss_temp, distill_loss_alpha)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

            optimizer.step()
            
            total_train_loss += loss.item()

            avg_loss = total_train_loss / (batch_idx + 1)

            pbar.set_description(f"Training... (avg. distillation loss {avg_loss:.3f})")

            scheduler.step()

        student.eval()

        total_val_loss = 0
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating..."):
            inputs = batch["input_ids"].to(device)
            
            student_logits = student(inputs[:, :-1], mask=student_mask).logits

            target_inputs = inputs[:, 1:]  # Shift right to create targets
            
            loss = torch.nn.functional.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)), target_inputs.reshape(-1), ignore_index=tokenizer.pad_token_id)
            
            total_val_loss += loss.item()

        epoch_grad_norms = util.compute_grad_norms(student)
        
        vanishing, exploding = util.detect_gradient_issues(epoch_grad_norms, vanish_thresh=1e-6, explode_thresh=10.0)

        vanishing_gradients = len(vanishing)
        exploding_gradients = len(exploding)

        print(f"Completed Epoch {epoch+1}: Average Train (distillation) Loss: {total_train_loss/len(train_loader):.4f}, Average Val (cross entropy) Loss: {total_val_loss/len(val_loader):.4f}")
        wandb.log({
            'epoch': epoch+1,
            'train-loss': total_train_loss/len(train_loader),
            'val_loss': total_val_loss/len(val_loader),
            'vanishing-gradients': vanishing_gradients,
            'exploding-gradients': exploding_gradients
        })
            
        if on_epoch_done is not None:
            on_epoch_done(epoch, student)

    wandb.finish()

    return student
