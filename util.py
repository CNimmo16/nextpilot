
import random

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