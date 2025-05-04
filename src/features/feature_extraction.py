import json
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

def build_processor(vocab_list, processor_save_path):
    """
    Create and save a Wav2Vec2Processor with a character-level tokenizer for Arabic.
    - `vocab_list`: List of characters (tokens) in the vocabulary (Arabic letters, space, etc.).
    - `processor_save_path`: Path to save the processor (feature extractor + tokenizer).
    """
    # Build vocabulary dict: character -> id
    vocab_dict = {token: idx for idx, token in enumerate(vocab_list)}
    # Write vocab to a JSON file
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False)
    # Initialize tokenizer with special tokens
    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token=" "
    )
    # Initialize feature extractor (for raw audio)
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True
    )
    # Combine into a processor and save
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer
    )
    processor.save_pretrained(processor_save_path)
    return processor
