# Utterance-to-Phoneme-Mapping-Using-RNNs--ASR-
# HW3P2: Utterance to Phoneme Mapping (ASR)

Course assignment (11-785, Spring 2026) on sequence-to-sequence automatic speech recognition. The model maps a variable-length sequence of speech feature vectors to a phoneme transcription using a recurrent encoder and CTC. Unlike HW1P2 (one phoneme label per frame), the target here is the natural, compressed phoneme sequence, which is order-aligned with the audio but not time-synchronous.

## Task

Given MFCC feature sequences of shape (T_in, 28), predict the phoneme transcription of length T_out, where T_out is much shorter than T_in and is not time-aligned to the frames. There are 40 phonemes (39 phonemes plus `[SIL]`) plus a BLANK symbol at index 0, for 41 output classes. Attention-based methods are disallowed; the encoder must be recurrent. Performance is the Levenshtein (character-level edit) distance between predicted and reference phoneme strings on the Kaggle test set; lower is better.

## Why CTC

The output is order-synchronous but time-asynchronous with the input, and the alignment between frames and phonemes is unknown. The two-stage solution has the network emit a probability vector at every (downsampled) time step, then a dynamic-programming layer derives the actual sequence. Training uses CTC loss, which computes the expected loss over all valid alignments via the forward-backward algorithm; the BLANK symbol lets the model emit genuine repeated phonemes and separate them after collapsing repeats. Inference collapses repeats and removes blanks, here via a beam-search CTC decoder.

## Data Pipeline

- **Source**: Kaggle competition `11785-hw-3-p-2-utterance-to-phoneme-mapping-spring-2026`. Partitions: `train-clean-100`, `dev-clean`, `test-clean`, each with an `mfcc/` folder and (for train/dev) a `transcripts/` folder.
- **Vocabulary**: a fixed ARPABET map (`CMUdict_ARPAbet`). `PHONEMES` indices map transcripts to integers; `LABELS` map predictions back to single-character phoneme symbols for edit-distance scoring. BLANK is index 0.
- **Dataset** (`AudioDataset`): loads MFCCs (28 features per frame), normalizes, strips `[SOS]`/`[EOS]` from transcripts, and maps phoneme strings to integer indices. No context window (unlike HW1P2). Files loaded in sorted order; test set must be the full set, never subsetted or shuffled.
- **collate_fn**: pads variable-length MFCCs and transcripts with `pad_sequence` and returns padded features, padded labels, and the unpadded lengths of each. Lengths are needed for packing and for CTC.
- **Augmentation** (training only): FrequencyMasking (param 15) and TimeMasking (param 35, applied twice), done in the collate function on the batch after permuting to (batch, freq, time), then permuted back.
- **DataLoaders**: train shuffled; validation and test not shuffled (test order must match Kaggle IDs).

## Model

An encoder-decoder ASR network (`ASRModel`) capped under the 20M trainable-parameter limit.

**Encoder** (`Encoder`)
- CNN embedding: three Conv1d layers (256 channels, kernel 3, stride 1), each followed by SiLU, BatchNorm1d, and dropout. `Permute` layers move between (batch, time, feat) and the (batch, channels, time) layout Conv1d expects.
- One bidirectional LSTM (hidden size 384).
- Two pyramidal BiLSTM (`pBLSTM`) layers, each halving the time resolution by concatenating adjacent timestep pairs (doubling feature dimension) before a 1-layer BiLSTM, for 4x total temporal downsampling.
- VariationalDropout after the first LSTM and after each pBLSTM.
- Throughout, sequences are packed (`pack_padded_sequence`) before each recurrent layer and unpacked (`pad_packed_sequence`) after, so padded regions do not contribute.

**Decoder** (`Decoder`)
- An MLP applied per timestep: Linear (2*embed -> 512), BatchNorm1d, SiLU, VariationalDropout, Linear (512 -> 41).
- LogSoftmax over the class dimension, then permuted to (time, batch, classes) as PyTorch CTCLoss expects.

The model returns log-probabilities plus the downsampled output lengths (needed by CTC and the decoder).

## Training Strategy

- **Loss**: `nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)`.
- **Optimizer**: AdamW (lr 1e-3, weight_decay 1e-4).
- **Scheduler**: ReduceLROnPlateau (mode min, factor 0.5, patience 5, min_lr 1e-6).
- **Mixed precision**: `torch.cuda.amp.GradScaler` / autocast.
- **Gradient clipping**: max_norm 2.0.
- Long training (the filled README notes ~200 epochs with learning-rate resets from the best checkpoint).

## Inference and Decoding

- **Decoder**: `torchaudio` CUDA CTC decoder (`cuda_ctc_decoder`) over the phoneme tokens. Beam width 1 during validation (to keep training fast) and 5 at test time.
- `decode_prediction` permutes logits to (batch, time, classes), guards against NaNs, runs the beam decoder, takes the top hypothesis per utterance, and maps token indices to phoneme label characters joined into a string. A CPU fallback (`decode_prediction_cpu`) is also provided.
- Validation tracks mean Levenshtein distance between decoded predictions and references.

## Config Summary

| Setting | Value |
|---|---|
| MFCC features | 28 |
| Embed / LSTM hidden size | 384 |
| pBLSTM layers | 2 (4x downsampling) |
| Output classes | 41 (40 phonemes + BLANK) |
| Batch size | 128 |
| Learning rate | 1e-3 (AdamW, wd 1e-4) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Dropout | 0.3 (encoder / lstm / decoder) |
| Train / test beam width | 1 / 5 |
| Augmentation | FreqMask 15, TimeMask 35 (x2) |
| Precision | mixed (AMP) |
| Epochs (config) | 150 |

## Running the Notebook

- **Environments supported**: Colab, Kaggle, and PSC Bridges-2, each with its own setup cells. Set `root` to the dataset location for your platform.
- **PSC note**: the notebook is configured for Bridges-2 with a V100-32GB GPU and the shared `IDLS26` conda environment; download the dataset to node-local `$LOCAL` storage to avoid shared-filesystem I/O bottlenecks (re-download per node).
- **Credentials**: requires a Kaggle API token (new `KGAT_` format) and, if logging is enabled, a WandB API key.
- **Checkpointing**: `save_model` / `load_model` persist model, optimizer, scheduler, metric, and epoch; training can resume from a checkpoint (`RESUME_TRAINING`).
- **Dependencies**: CUDA CTC decoding requires a GPU-enabled `torchaudio`.

## Submission Deliverables

- `submission.csv` of decoded phoneme strings for the Kaggle leaderboard (checkpoint cutoff: Levenshtein distance 12 or below).
- A completed `README` cell, the model metadata file, and the cleaned notebook, packaged into the Autolab submission zip.

## Rules

No pretrained or pre-built models; all modules (pBLSTM, encoder, decoder) implemented from base PyTorch. No external data. No attention-based techniques. Maximum 20M parameters including ensembles. Test set must be used in full and not shuffled.
