import argparse
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

def transcribe(model_dir, audio_path, device="cpu"):
    """
    Load a trained model and run ASR on a single audio file.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_dir)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
    model.eval()

    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.squeeze()
    target_sr = processor.feature_extractor.sampling_rate
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Prepare input
    input_values = processor(waveform.numpy(), sampling_rate=target_sr, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    return transcription

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned ASR model.")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to trained model and processor directory.")
    parser.add_argument("--audio_path", type=str, required=True,
                        help="Path to the input audio file.")
    args = parser.parse_args()
    result = transcribe(args.model_dir, args.audio_path)
    print("Transcription:", result)
