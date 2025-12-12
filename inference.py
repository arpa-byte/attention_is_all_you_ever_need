import torch
import spacy

from transformer.model.transformer import Transformer
from data_utils import get_dataloaders # We use this to get the vocabularies

def translate_sentence(sentence, src_field, trg_field, model, device, max_len=50):
    """
    Translates a single source sentence into the target language.

    Args:
        sentence (str): The source sentence to translate.
        src_field (Field): The source language Field object (contains vocab).
        trg_field (Field): The target language Field object (contains vocab).
        model (Transformer): The trained Transformer model.
        device: The device to run the model on (e.g., 'cpu' or 'cuda').
        max_len (int): The maximum length for the output sentence.

    Returns:
        str: The translated sentence.
    """
    model.eval()  # Set the model to evaluation mode

    # --- 1. Preprocess the source sentence ---
    # Tokenize and add <sos> and <eos> tokens
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    
    # Convert tokens to numerical indices
    src_indices = [src_field.vocab.stoi[token] for token in tokens]
    
    # Convert to a tensor and add a batch dimension
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    # --- 2. Run the Encoder ---
    # Create the source mask and encode the source sentence
    src_mask = model.make_src_mask(src_tensor) # We need to add this helper method to the model
    
    with torch.no_grad():
        enc_src = model.encoder(model.src_embedding(src_tensor), src_mask)

    # --- 3. Autoregressive Decoding Loop ---
    # Start the target sequence with the <sos> token
    trg_indices = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor) # And this one too
        
        with torch.no_grad():
            output, attention = model.decoder(model.trg_embedding(trg_tensor), enc_src, trg_mask, src_mask)
        
        # Get the predicted token for the last position
        pred_token = output.argmax(2)[:, -1].item()
        trg_indices.append(pred_token)

        # If the predicted token is <eos>, stop decoding
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
            
    # --- 4. Post-process the output ---
    # Convert the output indices back to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indices]
    
    # Return the translated sentence, removing the <sos> and <eos> tokens
    return " ".join(trg_tokens[1:-1])


if __name__ == '__main__':
    # --- 1. Load Model and Data Utilities ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 128 # This doesn't matter for inference, but needed by the function
    
    print("Loading vocabularies...")
    # We only need the vocab objects (SRC, TRG), not the iterators
    _, _, SRC, TRG = get_dataloaders(DEVICE, BATCH_SIZE)
    
    # --- 2. Define Hyperparameters and Instantiate Model ---
    # These MUST match the parameters of the trained model
    src_vocab_size = len(SRC.vocab)
    tgt_vocab_size = len(TRG.vocab)
    pad_idx = TRG.vocab.stoi[TRG.pad_token]
    D_MODEL = 512
    NUM_LAYERS = 3 # Must match the trained model
    H = 8
    D_FF = 2048
    DROPOUT = 0.1
    
    print("Instantiating model...")
    model = Transformer(src_vocab_size, tgt_vocab_size, D_MODEL, NUM_LAYERS, H, D_FF, DROPOUT, pad_idx).to(DEVICE)

    # --- 3. Load the Trained Weights ---
    # This assumes 'best-model.pt' exists in the root directory.
    # Your friend will provide you with this file after training is complete.
    try:
        print("Loading trained model weights...")
        model.load_state_dict(torch.load('best-model.pt', map_location=DEVICE))
    except FileNotFoundError:
        print("Error: 'best-model.pt' not found. Please train the model first.")
        exit()

    # --- 4. Translate a Sample Sentence ---
    # This is an example from the validation set
    src_sentence = "ein mann in einem orangefarbenen hut, der etwas anstarrt."
    
    print(f"\nSource Sentence: {src_sentence}")
    
    # Get the ground truth translation for comparison
    # (This is just for demonstration; in a real scenario you wouldn't have this)
    ground_truth_translation = "a man in an orange hat starring at something."
    print(f"Ground Truth Translation: {ground_truth_translation}")

    # Generate the model's translation
    predicted_translation = translate_sentence(src_sentence, SRC, TRG, model, DEVICE)
    print(f"Predicted Translation: {predicted_translation}")