import torch
from model import get_llama, SimpleDecoder
from tokenizer import tokenizer

def generate_code(
    model,
    tokenizer,
    prompt,
    max_length=100,
    device="cuda"
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(generated_ids)
            logits = outputs.logits
            logits = logits[:, -1, :]  # Get logits for the last token

            next_token = logits.argmax(dim=-1)

            # Append the generated token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code

if __name__ == '__main__':

    epoch = input('Enter an epoch to load weights for, or leave blank to use only the pretrained llama model: ')

    llama = get_llama()

    if epoch:
        model = SimpleDecoder(vocab_size=len(tokenizer)).to("cuda")

        checkpoint = torch.load(f"data/weights/nextjs_decoder_epoch_{epoch}.pth", weights_only=False)
        state_dict = checkpoint['model_state_dict']
        for key in list(state_dict.keys()):
            if key.startswith("module."):
                state_dict[key[len("module."):]] = state_dict.pop(key)

        model.load_state_dict(state_dict)

    # Define a prompt
    prompt = """
// File: src/components/ui/input.tsx
import * as React from "react"

import { cn } from "@/lib/utils"

export interface InputProps
extends React.InputHTMLAttributes<HTMLInputElement> {}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
({ className
"""

    # Generate code
    llama_code = generate_code(
        llama,
        tokenizer,
        prompt,
        max_length=100
    )

    # Generate code
    model_code = generate_code(
        model,
        tokenizer,
        prompt,
        max_length=100
    )

    print('Llama completion:')
    print(llama_code)

    print('\nDistilled model completion:')
    print(model_code)

