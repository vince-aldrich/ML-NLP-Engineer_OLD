from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, ElectraTokenizerFast, ElectraConfig, ElectraForSequenceClassification
import torch

def main():
    dataset = load_from_disk("data/tokenized_tweeteval")
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    config = ElectraConfig.from_pretrained("google/electra-small-discriminator", num_labels=3)
    model = ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator", config=config)

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="models/electra-finetuned-sentiment",
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        logging_dir="logs"
    )

    def compute_metrics(eval_pred):
        preds = torch.tensor(eval_pred.predictions).argmax(dim=1)
        labels = torch.tensor(eval_pred.label_ids)
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model("models/electra-finetuned-sentiment")
    tokenizer.save_pretrained("models/electra-finetuned-sentiment")
    print("✅ Training complete, model saved to models/electra-finetuned-sentiment")

if __name__ == "__main__":
    main()
