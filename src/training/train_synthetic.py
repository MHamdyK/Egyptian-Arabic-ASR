import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from dataclasses import dataclass
from typing import List, Union, Optional

from src.features.feature_extraction import build_processor
from src.utils.data_loader import load_dataset
from src.utils.metrics import compute_wer

@dataclass
class DataCollatorCTCWithPadding:
    """
    Collate function to pad inputs and labels for CTC training.
    Dynamically pads input_features and labels to the longest sequence in the batch.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None

    def __call__(self, features: List[dict]) -> dict:
        # Split into input and label dicts
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        # Pad inputs
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Pad labels (as targets) with the processor's tokenizer
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                return_tensors="pt"
            )
        # Replace padding token id's with -100 for CTC loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic speech data."""
    def __init__(self, csv_file, processor):
        self.audio_paths, self.transcripts = load_dataset(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        # Load audio file (assuming mono-channel WAV/FLAC)
        speech, sr = torch.load(path) if path.endswith('.pt') else None, None  # placeholder
        import torchaudio  # to avoid global import if not needed
        waveform, sr = torchaudio.load(path)
        waveform = waveform.squeeze()
        # Resample if needed
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        # Extract input values
        input_values = self.processor(waveform.numpy(), sampling_rate=target_sr).input_values[0]
        # Encode transcript as label IDs
        with self.processor.as_target_processor():
            labels = self.processor(self.transcripts[idx]).input_ids
        return {"input_values": torch.tensor(input_values, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int32)}

def train(args):
    # Build or load processor (feature extractor + tokenizer)
    vocab_list = [
        "ا","ب","ت","ث","ج","ح","خ","د","ذ","ر",
        "ز","س","ش","ص","ض","ط","ظ","ع","غ","ف",
        "ق","ك","ل","م","ن","ه","و","ي","ء"," ",
    ]
    processor = build_processor(vocab_list, args.model_dir)

    # Initialize model config and model (from scratch, random init)
    config = Wav2Vec2Config(vocab_size=len(vocab_list))
    model = Wav2Vec2ForCTC(config)
    model.to(args.device)

    # Prepare dataset and dataloader
    train_dataset = SyntheticDataset(args.synthetic_csv, processor)
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=data_collator)

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    model.train()
    loss_history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(args.device) for k,v in batch.items()}
            outputs = model(input_values=batch["input_values"], labels=batch["labels"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model and processor for reuse
    model.save_pretrained(args.model_dir)
    processor.save_pretrained(args.model_dir)
    print(f"Pre-training complete. Model saved to {args.model_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train Wav2Vec2 on synthetic Egyptian Arabic")
    parser.add_argument("--synthetic_csv", type=str, required=True,
                        help="Path to synthetic data CSV (path,transcript).")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory to save the pretrained model and processor.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Computation device (cuda or cpu).")
    args = parser.parse_args()
    train(args)
