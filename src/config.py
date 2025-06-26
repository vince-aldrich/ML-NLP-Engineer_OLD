from transformers import ElectraTokenizerFast, ElectraConfig

# Model + Tokenizer Name
MODEL_NAME = "google/electra-small-discriminator"

# Dataset name and subset
DATASET_NAME = "tweet_eval"
DATASET_SUBSET = "sentiment"

# Label count
NUM_LABELS = 3  # negative, neutral, positive

# Tokenizer
TOKENIZER = ElectraTokenizerFast.from_pretrained(MODEL_NAME)

# Config
MODEL_CONFIG = ElectraConfig.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Paths
TOKENIZED_DATA_PATH = "data/tokenized_tweeteval"
MODEL_SAVE_PATH = "models/electra-finetuned-sentiment"
LOG_DIR = "logs"
