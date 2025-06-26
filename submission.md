## Task Summary

The task was to fine-tune a pre-trained ELECTRA-Small model for text classification using the TweetEval benchmark dataset. The aim was to correctly categorize tweets into sentiment or emotion labels using only CPU-based training and inference in a clean, modular, and reproducible codebase.


## Approach

1. Dataset:

   * Used the TweetEval benchmark (via Hugging Face Datasets).
   * Focused on a subtask like Sentiment Classification (positive, negative, neutral).
   * Performed train/validation/test splits and stored in `data/`.

2. Model:

   * Used `google/electra-small-discriminator` as the base model.
   * Wrapped with `ElectraForSequenceClassification` for classification.
   * Adjusted config to match the number of labels and task.

3. Tokenization:

   * Used `ElectraTokenizerFast` to tokenize tweet text.
   * Enabled `truncation=True` and `padding=True`.

4. Training:

   * Used PyTorch and Hugging Face `Trainer` API.
   * TrainingArguments included:

     * `per_device_train_batch_size=16`
     * `evaluation_strategy="epoch"`
     * `save_total_limit=1` to keep minimal checkpoints

5. Evaluation:

   * Evaluated using accuracy and optionally F1 score.
   * Logged metrics per epoch.


Model Decisions

* Why ELECTRA-Small?

  * Faster and more compute-efficient than BERT due to its *discriminative training*.
  * Ideal for CPU-only environments and quick fine-tuning.
  * Smaller footprint (\~14M parameters) suitable for constrained devices.

* Why TweetEval?

  * Real-world social media text with emotion, hate, irony, and sentiment tasks.
  * Standardized benchmark for evaluating model performance on tweets.

* Trainer API Advantage:

  * Enables modular training and clean integration with metrics, logging, and saving.


##  Key Learnings

* ELECTRA's discriminative approach allows smaller models to match or beat larger BERT-like models on many NLP tasks.
* Working with TweetEval required cleaning and managing noisy real-world text (like hashtags, mentions, emojis).
* Using Trainer API significantly reduced boilerplate and helped isolate evaluation logic from model architecture.
* CPU-only training is slow, but feasible for small models like ELECTRA-Small.
