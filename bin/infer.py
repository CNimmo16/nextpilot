from inference import generate_code
import torch
from termcolor import colored
from tokenizer import tokenizer
from model import SimpleDecoder, get_llama

epoch = input('Enter an epoch to load weights for, or leave blank to use only the pretrained llama model: ')

llama = get_llama()

if epoch:
    model = SimpleDecoder(vocab_size=len(tokenizer)).to("cuda")

    checkpoint = torch.load(f"data/weights/epoch_{epoch}.pth", weights_only=False)
    state_dict = checkpoint['model_state_dict']
    for key in list(state_dict.keys()):
        if key.startswith("module."):
            state_dict[key[len("module."):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)

while True:

    prompt = input('Enter a prompt, or leave blank to use test prompt: ')

    if not prompt:
        prompt = """
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

    print(colored('Llama completion:', 'green'))
    print(f"{prompt}{colored(llama_code, 'yellow')}")

    print('\n')

    print(colored('Distilled model completion:', 'green'))
    print(f"{prompt}{colored(model_code, 'yellow')}")
