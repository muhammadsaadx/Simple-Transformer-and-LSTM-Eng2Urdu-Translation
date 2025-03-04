# English-to-Urdu Translation using Transformer and LSTM

This repository contains an implementation of an English-to-Urdu translation model using both Transformer and LSTM architectures. The notebook trains and evaluates these models on the UMC005 dataset, demonstrating their effectiveness in sequence-to-sequence translation.

## Features
- Implements both Transformer and LSTM-based translation models.
- Uses PyTorch and Hugging Face's `torchtext` and `tokenizers` for text processing.
- Trains on the UMC005 dataset.
- Provides evaluation metrics to compare model performance.
- Generates translated Urdu sentences from English input.

## Dataset
The UMC005 dataset is used for training the models. The dataset consists of English-Urdu sentence pairs and is preprocessed for tokenization and batching.

## Model Architectures

### Transformer Model
The Transformer model is based on the original implementation introduced in the paper *Attention is All You Need* by Vaswani et al. It consists of:
- **Multi-head self-attention mechanism**: Allows the model to focus on different parts of the input sentence simultaneously.
- **Positional encodings**: Helps the model understand word order, as Transformers do not inherently encode sequential information.
- **Encoder-decoder structure**: The encoder processes the input sequence, and the decoder generates the translated output by attending to the encoder’s representations.
- **Layer normalization**: Stabilizes training and improves convergence.

### LSTM Model
The LSTM-based model follows the standard sequence-to-sequence architecture with:
- **Bidirectional LSTM encoder**: Captures contextual dependencies in both forward and backward directions.
- **LSTM decoder with attention mechanism**: Enables the decoder to selectively focus on relevant parts of the input sequence while generating translations.
- **Teacher forcing during training**: Improves convergence by using the correct previous token as input instead of the model's predicted token during training.

## Training Process
The training pipeline consists of:
1. **Data Preprocessing**: 
   - Tokenization of English and Urdu sentences.
   - Creating vocabulary and mapping words to numerical tokens.
   - Padding sequences to ensure uniform input sizes.
2. **Model Training**:
   - Cross-entropy loss with label smoothing is used to improve generalization.
   - The Adam optimizer with learning rate scheduling is applied for efficient optimization.
   - Mini-batch training is performed using GPU acceleration if available.
3. **Validation & Monitoring**:
   - The BLEU score is used to evaluate translation quality.
   - Periodic validation is conducted to prevent overfitting.

## Evaluation
The models are assessed using:
- **BLEU Score**: Measures how closely the translated output matches human-generated references.
- **Qualitative Analysis**: Sample translations are examined to gauge fluency and coherence.

## Usage
After training, the models can be used to translate English sentences into Urdu. The trained models:
- Take an English sentence as input.
- Generate an Urdu translation using a probabilistic decoding strategy such as beam search.
- Can be further fine-tuned or extended for other sequence-to-sequence tasks.

