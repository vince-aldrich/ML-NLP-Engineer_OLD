## Model Architecture, Training Insights & Improvements

1. **Model Choice and Justification**
   I selected the `google/electra-small-discriminator` model for this sentiment classification task on TweetEval. The main reason was its computational efficiency — ELECTRA-Small is significantly faster and lighter than models like BERT, making it a great fit for environments like GitHub Codespaces that don’t have GPUs. Also, since ELECTRA uses a different pretraining strategy — replaced token detection rather than masked language modeling — it tends to learn better representations with fewer resources.

2. **Architecture Overview**
   The architecture is based on the ELECTRA-small backbone with a simple classification head added for the three sentiment classes: negative, neutral, and positive. I used the Hugging Face `ElectraForSequenceClassification` wrapper, which automatically includes the necessary linear layer on top. For tokenization, I used the `ElectraTokenizerFast` with padding and truncation set to a maximum length of 128 tokens to handle tweet-sized inputs efficiently.

3. **Training Configuration**
   I fine-tuned the model using Hugging Face’s `Trainer` API. Since this was all running on CPU inside a GitHub Codespace, I kept the batch sizes small and the number of epochs low to ensure reasonable training times. Specifically, I trained for 2 epochs with a learning rate of 2e-5, a training batch size of 16, and an evaluation batch size of 32. Evaluation was done at the end of every epoch. I also skipped saving intermediate checkpoints to reduce disk I/O and overhead.

4. **Training Outcomes and Observations**
   The training process was relatively smooth. The model was able to converge quickly — training loss dropped to around 0.35, and validation loss stabilized at approximately 0.45. The final validation accuracy landed around 80%, which is quite solid for a small model trained on CPU with limited tuning. There were no signs of overfitting in the learning curves across the two epochs.

5. **What Worked Well**
   The model was small enough to train within GitHub Codespaces without any memory issues or crashes. The predictions made sense, and the training time per epoch remained manageable even on CPU. The tokenizer also handled tweet-specific quirks like emojis, hashtags, and user handles pretty well.

6. **What Could Be Improved**
   Training on CPU is a major limitation — using a GPU would drastically reduce training time and allow for deeper hyperparameter exploration. I also didn’t experiment with learning rate schedulers, warmup steps, or dropout adjustments. These are obvious next steps for further improvement. Moreover, comparing ELECTRA-Small against tweet-specific models like `bertweet-base` or trying even lighter models like `TinyBERT` could provide valuable performance trade-offs.

7. **Final Thoughts**
   Overall, ELECTRA-Small proved to be a reliable and efficient choice for sentiment classification, especially in a resource-constrained environment like GitHub Codespaces. The balance between speed and accuracy was impressive, and the model is flexible enough to be adapted for other tweet-related classification tasks in the future.


