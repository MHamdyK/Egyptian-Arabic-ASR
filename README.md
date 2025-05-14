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
  
We use **Wav2Vec2-XLS-R (300M)** as our ASR backbone. Specifically:

- The model is initialized from the **pre-trained checkpoint** `facebook/wav2vec2-xls-r-300m`, leveraging transfer learning to enhance performance on our specific dataset.  
- **Fine-tuning Configuration:**  
  - **Attention Dropout:** 0.0  
  - **Hidden Dropout:** 0.0  
  - **Feature Projection Dropout:** 0.0  
  - **Time Masking Probability:** 0.05  
  - **Layer Dropout:** 0.0  
- The output layer is a **CTC head** (Connectionist Temporal Classification) with mean reduction, used for character prediction.  
- The model is configured to handle **shape mismatches** for the final layer using the `ignore_mismatched_sizes=True` argument.  
- **Vocabulary Size:** Matches the tokenizer used in training, dynamically obtained using `len(processor.tokenizer)`.  
- The **pad token ID** is set according to the tokenizer to ensure compatibility with input formatting.  
- The model is fine-tuned directly on **labeled audio data**, rather than performing self-supervised pre-training from scratch, as we leverage the robust feature extraction of the pre-trained XLS-R model.  

This setup ensures that the model can efficiently adapt to our specific dialect data while leveraging the strengths of a large-scale pre-trained architecture.


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
