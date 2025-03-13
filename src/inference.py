import torch

def generate_code(
    model,
    tokenizer,
    prompt,
    max_length=100,
    device="cpu"
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
            print('next token', next_token.item())

            # Append the generated token
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

            # Stop if the end-of-sequence token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence
    generated_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_code[len(prompt):]
