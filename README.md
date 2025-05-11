# Egyptian Arabic ASR Project

## Project Overview

This project builds an end-to-end Automatic Speech Recognition (ASR) system for Egyptian Arabic using a **Wav2Vec2** architecture. We follow a two-stage training pipeline:

1. **Pre-training from Scratch on Synthetic Data:** We generate a large synthetic Egyptian-Arabic speech corpus using OpenAI’s GPT-4o to create text transcripts, then convert text to speech via TTS. The Wav2Vec2 model is initialized from random weights and pre-trained on this synthetic data to learn acoustic representations.
2. **Fine-tuning on Real YouTube Data:** The model is further fine-tuned on real speech data scraped from YouTube videos in Egyptian Arabic. We utilize a publicly released Egyptian dialect audio dataset, where YouTube subtitles serve as transcripts.

Throughout, we use a shared Wav2Vec2 processor (feature extractor + tokenizer) for consistency. The processor is saved after pre-training and reused during fine-tuning and inference, ensuring identical normalization and tokenization pipelines.

## Dataset Description

- **Synthetic Dataset:** A custom pipeline (inspired by [KotP’s project](https://github.com/yousefkotp/Egyptian-Arabic-ASR-and-Diarization)) generates diverse Egyptian Arabic sentences using GPT-4o. These sentences are then fed to a TTS system (e.g., Fish-Speech) to create audio. The result is a large synthetic speech corpus with perfect transcripts.

- **Real Dataset (YouTube):** We use the *Egyptian Audio Dataset Collected From YouTube* (Ahmed Shafiq, Kaggle). This dataset contains audio snippets of various Egyptian Arabic speakers. Each video’s subtitles or captions are used as ground-truth transcripts:contentReference. The data covers multiple channels and topics for dialect diversity.

## Preprocessing

- **Audio Processing:** All audio is resampled to 16 kHz. We apply Wav2Vec2’s built-in normalization. 
- **Text Processing:** Transcripts are tokenized at the character level. We build a vocabulary of Arabic characters (letters, space, etc.) and initialize a `Wav2Vec2CTCTokenizer`. This vocabulary is saved and loaded via the Wav2Vec2 processor. By saving the processor, **the exact same feature extraction and tokenization is applied during training and inference**.
- **Data Collation:** Since audio/transcript lengths vary, we use a custom collator (`DataCollatorCTCWithPadding`) that pads input features and label sequences per batch. Padding positions in labels are set to -100 to be ignored by CTC loss.

## Model Architecture

We use **Wav2Vec2 (CTC)** as our ASR backbone. Specifically:

- The model is initialized **from scratch** (i.e., without pre-trained weights). The architecture is defined by `Wav2Vec2Config` with `vocab_size` matching our tokenizer.
- The convolutional feature extractor and Transformer encoder layers follow the standard Wav2Vec2 design.
- The output is connected to a CTC head that predicts characters.  
- Wav2Vec2’s self-supervised training (masking and contrastive loss) is *not* separately applied here; instead, we directly train on labeled audio via CTC. This effectively uses synthetic labeled data as a proxy for unlabeled pre-training.

## Training & Fine-tuning Pipeline

1. **Pre-Training on Synthetic Data (`train_synthetic.py`):**
   - Load synthetic dataset (CSV of audio paths and transcripts).
   - Build processor and model (randomly initialized).
   - Train with CTC loss for several epochs, logging training loss per epoch.
   - Save the model & processor (processor ensures shared preprocessing for later).
   
2. **Fine-Tuning on Real YouTube Data (`fine_tune.py`):**
   - Load the saved model and processor.
   - Load the YouTube audio dataset in the same format.
   - Continue training (CTC) on this real data for a few epochs.
   - Save the updated model.

The training scripts use PyTorch and Hugging Face Transformers. We employ best practices like learning rate scheduling and gradient clipping as needed. The code is modular and parameterized via command-line arguments for flexibility.

## Evaluation

After fine-tuning, we evaluate the model on a held-out test set of Egyptian Arabic. The primary metric is **Word Error Rate (WER)**. In our experiments, the model achieves around **49% WER** on test data, which is competitive for this low-resource dialect. (For reference, Wav2Vec2 has been shown to outperform previous ASR methods with limited data when properly trained.



Evaluation is done without an external language model (pure acoustic model decoding). We use `jiwer` to compute WER on the decoded transcripts.

## Inference Instructions

To transcribe new audio with the trained model:

1. Ensure `model_dir` contains the saved model and processor.
2. Run the inference script with a WAV or FLAC file:  
   ```bash
   python src/inference/infer.py --model_dir path/to/model --audio_path path/to/audio.wav
