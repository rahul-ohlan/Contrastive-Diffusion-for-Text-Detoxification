from src.utils.custom_tokenizer import create_tokenizer
from src.utils.data_utils_sentencepiece import get_dataloader
import torch

def main():
    # Load the tokenizer
    tokenizer = create_tokenizer(
        return_pretokenized=False,
        path="data/simple/"
    )
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")

    # Create dataloader
    batch_size = 2
    max_seq_len = 64
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        data_path="data/simple/simple.txt",
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )

    # Get and inspect a few batches
    print("\nTesting batches:")
    for i, (_, batch) in enumerate(dataloader):
        if i >= 3:  # Look at first 3 batches
            break
            
        print(f"\nBatch {i+1}:")
        print(f"Input shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        
        # Decode and print the first sequence in the batch
        input_ids = batch['input_ids'][0].tolist()
        decoded = tokenizer.decode(input_ids)
        print(f"First sequence decoded: {decoded}")
        
        # Print the attention mask for the first sequence
        attention = batch['attention_mask'][0].tolist()
        print(f"Attention mask: {attention[:10]}... (showing first 10 values)")
        
        # Print the actual tokens
        tokens = [tokenizer.decode([id]) for id in input_ids if id != 0]  # Skip padding tokens
        print(f"Tokens: {tokens}")

if __name__ == "__main__":
    main() 