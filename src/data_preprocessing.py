from datasets import load_dataset, DatasetDict
from transformers import ElectraTokenizerFast

def preprocess_data(tokenizer: ElectraTokenizerFast) -> DatasetDict:
    dataset = load_dataset("tweet_eval", "sentiment")
    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)
    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized

def main():
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    tokenized_data = preprocess_data(tokenizer)
    tokenized_data.save_to_disk("data/tokenized_tweeteval")
    print("✅ Data preprocessing done. Saved at data/tokenized_tweeteval")

if __name__ == "__main__":
    main()
