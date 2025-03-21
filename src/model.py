import torch
from transformers import BitsAndBytesConfig, LlamaForCausalLM
from src.tokenizer import tokenizer
from types import SimpleNamespace

class SimpleDecoder(torch.nn.Module):
    def __init__(self, vocab_size=10000, max_seq_len=512):
        super().__init__()
        # Hyperparameters controlling model size
        self.d_model = 480       # Embedding dimension
        self.n_layers = 4        # Number of decoder layers
        self.n_heads = 8         # Attention heads
        self.d_ff = 1920         # Feed-forward dimension\

        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embed = torch.nn.Embedding(vocab_size, self.d_model)
        self.pos_embed = torch.nn.Embedding(max_seq_len, self.d_model)
        
        # Decoder layers
        self.layers = torch.nn.ModuleList([
            DecoderLayer(self.d_model, self.n_heads, self.d_ff)
            for _ in range(self.n_layers)
        ])
        
        # Final projection
        self.proj = torch.nn.Linear(self.d_model, vocab_size)

        torch.nn.init.xavier_uniform_(self.proj.weight)
        
        # Count parameters (~20M)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params/1e6:.2f}M")

    def forward(self, x, mask=None):
        b, t = x.size()
        positions = torch.arange(0, t, dtype=torch.long, device=x.device)
        x = self.token_embed(x) + self.pos_embed(positions)
        
        for layer in self.layers:
            x = layer(x, mask)

        logits = self.proj(x)
            
        return SimpleNamespace(logits=logits)

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # Self-attention
        self.self_attn = torch.nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_norm = torch.nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = torch.nn.LayerNorm(d_model)

        torch.nn.init.xavier_uniform_(self.ffn[0].weight)  # First Linear layer in FFN
        torch.nn.init.xavier_uniform_(self.ffn[2].weight)  # Second Linear layer in FFN

    def forward(self, x, mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.attn_norm(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        return x

def get_llama(device = 'cpu'):
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    except Exception:
        bnb_config = None

    llama = LlamaForCausalLM.from_pretrained(
        "codellama/CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    ).to(device)
    # resize to account for added pad token
    llama.resize_token_embeddings(len(tokenizer))

    return torch.nn.DataParallel(llama)
