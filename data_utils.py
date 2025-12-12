import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import os

def get_dataloaders(device, batch_size: int):
    """
    Creates the data iterators for the Multi30k dataset using the legacy torchtext API.
    This version explicitly specifies the filenames to match our manually downloaded data.

    Args:
        device: The device to place the tensors on.
        batch_size (int): The number of sequences per batch.

    Returns:
        tuple: A tuple containing train/valid iterators and the SRC/TRG Field objects.
    """
    # --- 1. Load SpaCy tokenizers ---
    try:
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        print("SpaCy models not found. Please run: python -m spacy download de_core_news_sm en_core_web_sm")
        exit()

    def tokenize_de(text):
        return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

    # --- 2. Define the Fields for processing the data ---
    SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
    TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

    # --- 3. Load the Multi30k dataset using the Fields ---
    # !!! THIS IS THE FIX !!!
    # We are now explicitly telling torchtext the filenames to look for,
    # overriding the incorrect defaults.
    train_data, valid_data, test_data = Multi30k.splits(
        exts=('.de', '.en'), 
        fields=(SRC, TRG),
        # Explicitly provide the filenames without extensions
        train='train',
        validation='val',
        test='test_2016_flickr'
    )

    # --- 4. Build vocabularies from the training data ---
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    # --- 5. Create Data Iterators ---
    train_iterator, valid_iterator, _ = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    return train_iterator, valid_iterator, SRC, TRG

# Standalone Test
if __name__ == '__main__':
    print("--- Running Standalone Test for Data Utilities (Legacy torchtext) ---")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    
    # This call will now succeed.
    train_iter, valid_iter, SRC, TRG = get_dataloaders(DEVICE, BATCH_SIZE)
    
    print(f"Source vocab size: {len(SRC.vocab)}")
    print(f"Target vocab size: {len(TRG.vocab)}")
    
    batch = next(iter(train_iter))
    src_batch = batch.src
    tgt_batch = batch.trg
    
    print("\nSample batch shapes:")
    print("Source batch shape:", src_batch.shape)
    print("Target batch shape:", tgt_batch.shape)
    
    assert src_batch.shape[0] <= BATCH_SIZE # BucketIterator can sometimes return a smaller last batch
    assert tgt_batch.shape[0] <= BATCH_SIZE
    
    print("\n--- Standalone Test Passed! ---")