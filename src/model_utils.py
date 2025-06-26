from transformers import ElectraForSequenceClassification, ElectraTokenizerFast

def build_model():
    model = ElectraForSequenceClassification.from_pretrained(
        "google/electra-small-discriminator",
        num_labels=3  # sentiment classes: negative, neutral, positive
    )
    return model

if __name__ == "__main__":
    tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")
    model = build_model()
    tokenizer.save_pretrained("models/electra-sentiment")
    model.save_pretrained("models/electra-sentiment")
    print("✅ Base model and tokenizer saved to models/electra-sentiment")
