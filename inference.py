import torch
from torch.nn.functional import softmax
from model import get_llama
from tokenizer import tokenizer

def generate_code(
    model,
    tokenizer,
    prompt,
    max_length=100,
    device="cuda"
):
    """
    Generate code completions using your model.

    Args:
        model: Your trained decoder model.
        tokenizer: The tokenizer used for encoding/decoding.
        prompt: The input prompt (string).
        max_length: Maximum length of the generated sequence.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling (limit sampling to top-k tokens).
        top_p: Nucleus sampling (limit sampling to top-p probability mass).
        device: Device to run the model on ("cuda" or "cpu").

    Returns:
        Generated code completion (string).
    """
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
    model = get_llama()

    # model = SimpleDecoder(vocab_size=tokenizer.vocab_size).to("cuda")

    # checkpoint = torch.load("data/weights/nextjs_decoder_epoch_3.pth", weights_only=False)
    # state_dict = checkpoint['model_state_dict']
    # for key in list(state_dict.keys()):
    #     if key.startswith("module."):
    #         state_dict[key[len("module."):]] = state_dict.pop(key)

    # model.load_state_dict(state_dict)

    # Define a prompt
    prompt = """
    // Next.js API route example
    export default function handler(req, res) {
    if (req.method === 'POST') {
    """

    # Generate code
    generated_code = generate_code(
        model,
        tokenizer,
        prompt,
        max_length=100
    )

    print(generated_code)
