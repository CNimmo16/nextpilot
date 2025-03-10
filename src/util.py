
import random
import torch

def generate_friendly_model_id(tokenizer, num_words=4):
    # Get vocabulary as a list
    vocab = list(tokenizer.get_vocab().keys())
    
    # Filter for actual words (usually longer tokens without special chars)
    word_candidates = [
        word for word in vocab 
        if len(word) >= 4  # Reasonable word length
        and word.isalpha()  # Contains only letters
        and not word.startswith('Ġ')  # Not a BPE prefix token
        and not word.startswith('▁')  # Not a Sentence Piece prefix
        and not word.startswith('<')  # Not a special token
    ]
    
    # Select random words
    selected_words = random.sample(word_candidates, min(num_words, len(word_candidates)))
    
    # Join with underscore and lowercase
    model_id = "_".join([word.lower() for word in selected_words])
    
    return model_id

def compute_grad_norms(model: torch.nn.Module):
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    return grad_norms

def detect_gradient_issues(grad_norms, vanish_thresh: int, explode_thresh: int):
    vanishing = {name: norm for name, norm in grad_norms.items() if norm < vanish_thresh}
    exploding = {name: norm for name, norm in grad_norms.items() if norm > explode_thresh}

    if vanishing:
        print(f'❗Vanishing gradients detected: {vanishing}')
    if exploding:
        print(f'❗Exploding gradients detected: {exploding}')

    return vanishing, exploding
