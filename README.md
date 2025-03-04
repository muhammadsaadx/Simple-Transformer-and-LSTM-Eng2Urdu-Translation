# English-to-Urdu Translation using Transformer and LSTM

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
- Multi-head self-attention mechanism.
- Positional encodings to retain word order.
- Encoder-decoder structure with layer normalization.

### LSTM Model
The LSTM-based model follows the standard sequence-to-sequence architecture with:
- Bidirectional LSTM encoder.
- LSTM decoder with attention mechanism.
- Teacher forcing during training to improve convergence.

## Training
The models are trained using:
- Cross-entropy loss with label smoothing.
- Adam optimizer with learning rate scheduling.
- GPU acceleration when available.

To train the models, simply run the notebook cells sequentially. Ensure the dataset is properly loaded before training.

## Evaluation
The models are evaluated using:
- BLEU score to measure translation quality.
- Sample translations for qualitative analysis.

## Usage
After training, you can use the models to generate translations by providing English sentences as input. The trained models output Urdu translations with probabilistic beam search decoding.

## License
This project is licensed under the MIT License.

