import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from src.utils.data_loader import load_dataset
from src.utils.metrics import compute_wer
import torchaudio

class RealDataset(Dataset):
    """Dataset for real Egyptian Arabic audio collected from YouTube."""
    def __init__(self, csv_file, processor):
        self.audio_paths, self.transcripts = load_dataset(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        waveform, sr = torchaudio.load(path)
        waveform = waveform.squeeze()
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        input_values = self.processor(waveform.numpy(), sampling_rate=target_sr).input_values[0]
        with self.processor.as_target_processor():
            labels = self.processor(self.transcripts[idx]).input_ids
        return {"input_values": torch.tensor(input_values, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int32)}

def fine_tune(args):
    # Load the pretrained model and processor
    processor = Wav2Vec2Processor.from_pretrained(args.model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_dir).to(args.device)

    # Prepare dataset
    real_dataset = RealDataset(args.real_csv, processor)
    data_collator = torch.utils.data.dataloader.default_collate  # we'll do simple collate
    real_loader = DataLoader(real_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=lambda x: data_collator(x))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        for batch in real_loader:
            # Collate_fn returns lists, so pad manually
            input_values = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["input_values"]) for item in batch],
                batch_first=True
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["labels"]) for item in batch],
                batch_first=True,
                padding_value=-100
            )
            input_values = input_values.to(args.device)
            labels = labels.to(args.device)

            outputs = model(input_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{args.epochs} done.")

    # Save fine-tuned model
    model.save_pretrained(args.model_dir)
    print(f"Fine-tuning complete. Model updated at {args.model_dir}.")

    # Basic evaluation (compute WER on a small held-out set or part of data)
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in real_loader:
            input_values = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(item["input_values"]) for item in batch],
                batch_first=True
            ).to(args.device)
            logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            # Decode predictions
            pred_strs = processor.batch_decode(pred_ids)
            # Decode references (strip padding)
            ref_strs = []
            for item in batch:
                label_ids = [token_id for token_id in item["labels"] if token_id != -100]
                ref_strs.append(processor.decode(label_ids))
            predictions.extend(pred_strs)
            references.extend(ref_strs)
    wer_score = compute_wer(predictions, references)
    print(f"Evaluation WER: {wer_score:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 on real Egyptian YouTube data")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of pretrained model/processor (from synthetic phase).")
    parser.add_argument("--real_csv", type=str, required=True,
                        help="Path to real data CSV (path,transcript).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    fine_tune(args)
