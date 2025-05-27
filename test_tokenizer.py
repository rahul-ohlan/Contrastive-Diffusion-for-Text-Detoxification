from src.utils.custom_tokenizer import create_tokenizer

# Test BERT tokenizer
tokenizer = create_tokenizer(return_pretokenized=True, path=".", tokenizer_type="word-level")

# Test some sample texts with different characteristics
test_texts = [
    "Hello world! This is a test sentence.",  # Basic sentence
    "BERT converts THIS to lowercase.",       # Case conversion
    "I love ML-based AI systems!",           # Hyphenated words
    "The price is $99.99 today.",            # Numbers and special characters
    "🤗 Hugging Face is awesome!"            # Emojis
]

for text in test_texts:
    encoded = tokenizer(text)
    decoded = tokenizer.decode(encoded['input_ids'])
    
    print("\nOriginal text:", text)
    print("Encoded tokens:", tokenizer.convert_ids_to_tokens(encoded['input_ids']))
    print("Encoded ids:", encoded['input_ids'])
    print("Decoded text:", decoded)

print("\nVocabulary size:", len(tokenizer)) 