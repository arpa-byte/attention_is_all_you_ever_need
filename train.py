import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm

from transformer.model.transformer import Transformer
from data_utils import get_dataloaders

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, iterator, optimizer, criterion, clip):
    """
    Performs one full training pass over the dataset.
    """
    model.train()  # Set the model to training mode (enables dropout)
    epoch_loss = 0

    # Use tqdm for a progress bar
    for batch in tqdm(iterator, desc="Training"):
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        # --- The Core Training Step ---
        # The model's input should be the target sequence without the final <eos> token.
        # The ground truth for the loss function should be the target sequence without the initial <sos> token.
        # This is because we are predicting the next word in the sequence.
        output = model(src, trg[:, :-1])
        
        # Reshape for CrossEntropyLoss
        # output shape: (batch_size, trg_len - 1, output_dim)
        # ground_truth shape: (batch_size, trg_len - 1)
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        ground_truth = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, ground_truth)
        loss.backward()
        
        # Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Measures the model's performance on the validation dataset.
    """
    model.eval()  # Set the model to evaluation mode (disables dropout)
    epoch_loss = 0
    
    with torch.no_grad():  # No need to calculate gradients during evaluation
        for batch in iterator:
            src = batch.src
            trg = batch.trg

            # Perform the forward pass with the shifted target sequence
            output = model(src, trg[:, :-1])
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            ground_truth = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, ground_truth)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    """Calculates the duration of an epoch."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

if __name__ == '__main__':
    # --- 1. Hyperparameters ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128 # Increased batch size for more stable training
    D_MODEL = 512
    NUM_LAYERS = 3 # Reduced layers for faster training in the assignment
    H = 8
    D_FF = 2048
    DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    N_EPOCHS = 10
    CLIP = 1.0
    
    print(f"Using device: {DEVICE}")

    # --- 2. Load Data ---
    print("Loading data...")
    train_iterator, valid_iterator, SRC, TRG = get_dataloaders(DEVICE, BATCH_SIZE)
    
    src_vocab_size = len(SRC.vocab)
    tgt_vocab_size = len(TRG.vocab)
    pad_idx = TRG.vocab.stoi[TRG.pad_token]
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")

    # --- 3. Initialize Model, Optimizer, and Loss Function ---
    print("Initializing model...")
    model = Transformer(src_vocab_size, tgt_vocab_size, D_MODEL, NUM_LAYERS, H, D_FF, DROPOUT, pad_idx).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    print(f"\nThe model has {count_parameters(model):,} trainable parameters.")

    # --- 4. Main Training Loop ---
    best_valid_loss = float('inf')

    print("\n--- Starting Training ---")
    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        # Save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {torch.exp(torch.tensor(train_loss)):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {torch.exp(torch.tensor(valid_loss)):7.3f}')
    
    print("\n--- Training Complete ---")