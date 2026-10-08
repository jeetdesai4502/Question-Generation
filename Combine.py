#!/usr/bin/env python
# coding: utf-8

# # SECTION 1: DESCRIPTIVE ANALYSIS

# In[1]:


import pandas as pd

# 1. Load Data for Exploratory Analysis

# useing lines=True because the dataset is in JSONL (newline-delimited JSON) format.
# this allows for line-by-line processing.
df_train = pd.read_json('qg_train.jsonl', lines=True)

# 2. Structural Inspection

# Checking the shape and data types to identify potential nulls or format issues early.
# this paty confirms haveing 9489 entries and no missing values (non-null).
print("--- Dataset Info ---")
df_train.info()

# 3. Content Preview

# adjusting the column width temporarily to see the full text content,
# ensuring there are no obvious encoding errors (like weird symbols) in the first few rows.
pd.set_option('display.max_colwidth', None)

print("\n--- First 5 Rows Preview ---")
print(df_train.head(5))


# 2.2 Sequence Length and Vocabulary Analysis
# 
# Before designing the model architecture, it is critical to determine two key hyperparameters: the maximum sequence length and the vocabulary size.
# 
# Sequence Length: Neural networks require fixed-size inputs. If the chosen length is too short, we truncate information; if too long, we waste memory on padding. We analyze the word count distribution to find the optimal balance.
# 
# Vocabulary Size: The number of unique tokens dictates the size of the embedding layer.
# 
# The following script calculates these statistics to inform our preprocessing strategy.

# In[2]:


# 4. Sequence Length Analysis

# We calculate the word count for each input text.
# This is crucial for determining the 'max_seq_length' hyperparameter later.
# If our model's limit is 512 but our average is only 25, we might waste memory
# or need to adjust our padding strategy.
df_train['word_count'] = df_train['text'].str.split().str.len()

# Display specific statistics to understand the distribution
print("--- Sentence Length Statistics ---")
print(df_train[['text', 'word_count']].head(5))
print("\nSummary Metrics:")
print(df_train['word_count'].describe()) # describe() gives mean, std, min, max in one go

# 5. Vocabulary Size Analysis

# We need to know the number of unique tokens to size our embedding layer correctly.
# Using .explode() is more efficient than a loop for creating a flat list of words.
all_words = df_train['text'].str.split().explode()
vocab_size = all_words.nunique()

print(f"\n--- Vocabulary Metrics ---")
print(f"Total Words (Tokens): {len(all_words)}")
print(f"Unique Vocabulary Size: {vocab_size} words")


# Analysis of Results
# 
# The statistics above reveal two important characteristics:
# 
# Short Input Sequences: The average text length is approximately 24.5 words. This suggests the data consists of sentence-level context. We can safely set a small MAX_SEQ_LENGTH (e.g., 64 or 128) for efficient training.
# 
# Vocabulary Size: The dataset contains approximately 30,000 unique words, which fits well with standard pre-trained tokenizers (like BERT) without needing aggressive vocabulary pruning.

# 2.3 Target Variable Analysis (Question Types)
# 
# In Question Generation, the type of question (e.g., What, Who, Why) dictates the cognitive difficulty.
# 
# Factoid (What/Who): Rely on entity extraction.
# 
# Reasoning (Why/How): Require understanding causality.
# 
# We analyze the distribution of interrogative words to check for class imbalance, which could bias the model generation.

# In[3]:


# 6. Question Type Distribution

# We extract the first word of each target question (the 'Interrogative' or 'WH-word').
# This helps us understand the complexity of the generation task.
# 'What/Who' questions usually require fact retrieval, while 'Why/How' require reasoning.
df_train['question_type'] = df_train['gold_label_str'].str.split().str[0]

# Calculate raw counts and relative frequencies (percentages)
type_counts = df_train['question_type'].value_counts()
type_percents = df_train['question_type'].value_counts(normalize=True) * 100

# Create a summary dataframe for cleaner printing
distribution_summary = pd.DataFrame({
    'Count': type_counts,
    'Percentage': type_percents.map('{:.1f}%'.format) # Format as 42.1%
})

print("--- Question Type Distribution (Top 7) ---")
print(distribution_summary.head(7))

# Analytical Note for Report:
# A dominance of 'What' and 'Who' suggests the dataset focuses heavily on
# entity extraction rather than complex reasoning.


# # SECTION 2: DATA PROCESSING

# 3. Data Integrity and Deduplication
# 
# Before training, we must ensure data reliability. A common issue in scraped datasets is duplicate entries.
# 
# Why this matters: If the training set contains duplicates, the model may "memorize" these examples rather than generalizing. This can also lead to Data Leakage if duplicates appear in the validation set. We will quantify and remove these duplicates to ensure a valid training distribution.

# In[4]:


# 7. Data Integrity Verification

# We check for exact duplicates (identical text and target).
# Presence of duplicates can lead to 'overfitting' where the model memorizes
# these specific examples instead of learning general patterns.
original_rows = len(df_train)
duplicate_count = df_train.duplicated().sum()
duplicate_percent = (duplicate_count / original_rows) * 100

print("--- Duplicate Analysis ---")
print(f"Original Row Count: {original_rows}")
print(f"Duplicates Found:   {duplicate_count} ({duplicate_percent:.2f}%)")

# Decision Logic:
# Since we found duplicates, we must remove them in the pre-processing stage
# to ensure a valid training distribution.


# Decision Logic: The analysis identified exact duplicates in the training data. To prevent overfitting, we will remove these redundant rows. Note that we do not remove duplicates from the Validation or Test sets, as those must remain exactly as provided to ensure a fair comparison with other models.

# In[5]:


# 8. Data Sanitization

# Based on our analysis above, we proceed to remove the identified duplicates.
# We overwrite 'df_train' with the clean version.
# inplace=False is default, so we re-assign to the variable.
df_train_clean = df_train.drop_duplicates()

# Calculate the difference to verify the operation
removed_count = len(df_train) - len(df_train_clean)

# Update the main dataframe variable
df_train = df_train_clean

print("--- Sanitization Complete ---")
print(f"Old Row Count: {original_rows}")
print(f"New Row Count: {len(df_train)}")
print(f"Dropped Rows:  {removed_count}")

# Note: The dataset is now ready for tokenization and model training.


# # SECTION 3: PRE-PROCESSING

# 4. Text Pre-processing and Normalization
# 
# Real-world text contains "noise" that adds no semantic meaning but increases vocabulary size. Our goal is to normalize the text so the model focuses on linguistic content.
# 
# Strategy: We define a cleaning function (clean_text) using Regular Expressions (Regex) to:
# 
# Remove HTML Artifacts: Strips tags like <br>.
# 
# Filter Non-Semantic Entities: We remove usernames (@user) and URLs entirely.
# 
# Rationale: These unique handles are irrelevant to the grammar of the question. Removing them reduces the vocabulary size and forces the model to focus on context words.

# In[6]:


import re

# 9. Text Pre-processing Logic

# We enable full column width temporarily to visually verify the cleaning results.
pd.set_option('display.max_colwidth', None)

def clean_text(text):
    """
    Applies regex-based cleaning to filter out non-semantic noise.

    Design Decision:
    We choose to remove handles and URLs entirely (rather than masking them)
    to prevent the model from focusing on irrelevant entities. This aggressive
    cleaning helps reduce the vocabulary size to only meaningful English words.
    """

    # 1. Remove HTML tags
    # Artifacts like <br> or <div> add no meaning to the sentence.
    text = re.sub(r'<[^>]+>', '', text)

    # 2. Remove Usernames (@user)
    # Specific user handles are not relevant for the general task of
    # question generation, so we treat them as noise and remove them.
    text = re.sub(r'@\w+', '', text)

    # 3. Remove URLs
    # Links are removed as they do not contribute to the semantic
    # context needed to generate questions.
    text = re.sub(r'http\S+', '', text)

    # 4. Whitespace Cleanup
    # Fix any double spaces created by the deletions above.
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Apply the cleaning function
# We create a new column 'clean_text' to validate the Before/After states
df_train['clean_text'] = df_train['text'].apply(clean_text)

# Validation: Print a comparison to ensure noise was removed correctly
print("--- Pre-processing Comparison (Noise Removal) ---")
print(df_train[['text', 'clean_text']].head())


# 5. Automated Processing Pipeline
# 
# Now that we have defined our cleaning logic, we combine the steps into a robust pipeline.
# 
# The following script automates the process for all three splits (train, validation, test). It ensures that duplicates are removed only from the Training set, while text normalization is applied consistently across all files.

# In[7]:


import pandas as pd
import re
import os

# 1. Cleaning Function (Noise Removal Strategy)
def clean_text(text):
    """
    Applies regex to strip non-semantic noise from the raw text.
    """
    # Remove HTML artifacts (e.g. <br>)
    text = re.sub(r'<[^>]+>', '', text)

    # Remove Usernames & Links
    # Design Decision: We remove these entirely (rather than masking) to
    # significantly reduce the vocabulary size and force the model to focus
    # only on the English words, not random entities.
    text = re.sub(r'@\w+', '', text)       # Delete @mentions
    text = re.sub(r'http\S+', '', text)    # Delete URLs

    # Clean up any leftover whitespace from the deletions
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 2. Automated Processing Pipeline
def process_and_save(filename, is_training_set=False):
    # Validates that the file exists before attempting to load it
    if not os.path.exists(filename):
        print(f"Skipping {filename} (File not found). Please upload it.")
        return

    print(f"Processing pipeline started for: {filename}...")

    # Load dataset
    df = pd.read_json(filename, lines=True)
    initial_count = len(df)

    # DEDUPLICATION (Training Only)
    # We remove duplicates from training data to prevent leakage/memorization.
    # Validation and Test sets are kept exactly as provided to ensure
    # fair comparison with other models.
    if is_training_set:
        df = df.drop_duplicates()
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"  - Data Integrity: Removed {dropped} duplicate rows.")

    # Apply the text cleaning logic defined above
    df['text'] = df['text'].apply(clean_text)

    # Save the processed version
    # naming convention '_cleaned' prevents overwriting raw data
    output_filename = filename.replace('.jsonl', '_cleaned.jsonl')
    df.to_json(output_filename, orient='records', lines=True)

    print(f"  - Pipeline Complete. Saved {len(df)} rows to {output_filename}\n")

# 3. Execution

# We map the filenames to a boolean flag indicating if they are training data
files_to_process = [
    ('qg_train.jsonl', True),       # True = Deduplicate this one
    ('qg_validation.jsonl', False), # False = Keep 100% intact
    ('qg_test.jsonl', False)        # False = Keep 100% intact
]

print("--- STARTING BATCH PROCESSING ---\n")
for file, is_train in files_to_process:
    process_and_save(file, is_training_set=is_train)
print("--- BATCH PROCESSING FINISHED ---")


# # # Machine learning algorithm: LSTM

# In[8]:


import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


# json: used to read dataset stored in JSONL format
# 
# Counter: helps count word frequencies for vocabulary building
# 
# torch: core deep learning library
# 
# Dataset and DataLoader: enable efficient batching and shuffling of data

# In[9]:


def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Load training and validation datasets
train_data = load_jsonl('qg_train_cleaned.jsonl')
vali_data   = load_jsonl('qg_validation_cleaned.jsonl')

print("Train size:", len(train_data))
print("Validation size:", len(vali_data))


# Each dataset item contains:
# 
# text: context paragraph
# 
# gold_label_str: target question
# 
# Splitting into train and validation sets helps evaluate generalization

# In[10]:


# Special tokens required for sequence-to-sequence learning
PADTOKEN = '<PAD>'   # Padding token
SOSTOKEN = '<SOS>'   # Start of sentence
EOSTOKEN = '<EOS>'   # End of sentence
UNKTOKEN = '<UNK>'   # Unknown word


# These tokens allow:
# 
# 1)Variable-length sequences
# 
# 2)Explicit start and end of generation
# 
# 3)Handling unseen words

# In[11]:


def build_vocab(data, min_freq=2):
    """
    Build vocabulary from training data.
    Words appearing less than min_freq times
    are discarded to reduce noise.
    """
    counter = Counter()
    # Count words from both context and questions
    for item in data:
        counter.update(item['text'].split())
        counter.update(item['gold_label_str'].split())
     # Initialize vocabulary with special tokens
    vocab = {
        PADTOKEN: 0,
        SOSTOKEN: 1,
        EOSTOKEN: 2,
        UNKTOKEN: 3
    }

    # Add frequent words to vocabulary
    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


# In[12]:


# Build vocabulary using training data only
vocab = build_vocab(train_data)
vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)


# Vocabulary is built only from training data to avoid data leakage
# 
# Low-frequency words are removed to:
# 
# 1)reduce overfitting
# 
# 2)reduce vocabulary size

# In[13]:


def encode_sentence(sentence, vocab, max_len, add_sos_eos=False):
    """
    Convert a sentence into a fixed-length tensor of word indices.
    """
    tokens = sentence.split()
    
    # Add <SOS> and <EOS> tokens if required
    if add_sos_eos:
        tokens = [SOSTOKEN] + tokens + [EOSTOKEN]
    # Convert tokens to indices (use <UNK> if unseen)
    ids = [vocab.get(tok, vocab[UNKTOKEN]) for tok in tokens]
    # Truncate or pad sequence to fixed length
    ids = ids[:max_len]
    ids += [vocab[PADTOKEN]] * (max_len - len(ids))

    return torch.tensor(ids)


# 1)Converts text into numerical form required by neural networks
# 
# 2)Ensures all sequences have the same length

# In[14]:


class QGDataset(Dataset):
    """
    Dataset class for Question Generation.
    """
    def __init__(self, data, vocab, max_input_len=50, max_output_len=20):
        self.data = data
        self.vocab = vocab
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Encoder input: context paragraph
        encoder_input = encode_sentence(
            item['text'], self.vocab, self.max_input_len
        )

        # Decoder output: target question (with SOS and EOS)
        decoder_output = encode_sentence(
            item['gold_label_str'], self.vocab,
            self.max_output_len, add_sos_eos=True
        )

        return encoder_input, decoder_output


# 1)Separates:
#     -encoder input (context)
#     -decoder target (question)
# 
# 2)Adds <SOS> and <EOS> only to the decoder side

# In[15]:


train_dataset = QGDataset(train_data, vocab)
vali_dataset   = QGDataset(vali_data, vocab)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
vali_loader   = DataLoader(vali_dataset, batch_size=32)


# shuffle=True improves training stability
# 
# Batching improves training efficiency

# In[16]:


import torch.nn as nn

class Encoder(nn.Module):
    """
    LSTM-based Encoder that encodes the input context
    into a fixed-size hidden representation.
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


# -Maps input tokens to embeddings
# 
# -LSTM summarizes entire sequence into final hidden and cell states

# In[17]:


class Decoder(nn.Module):
    """
    LSTM-based Decoder that generates the question
    token-by-token.
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden, cell


# In[18]:


class Seq2Seq(nn.Module):
    """
    Combines Encoder and Decoder into a single
    Seq2Seq model.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        # Encode input sequence
        hidden, cell = self.encoder(src)
        # First input to decoder is <SOS>
        x = trg[:, 0]  # <SOS>

        # Teacher forcing
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[:, t] = output
            x = trg[:, t]  # teacher forcing

        return outputs


# In[19]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(vocab_size, 100, 256)
decoder = Decoder(vocab_size, 100, 256)
model_lstm = Seq2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab[PADTOKEN])
optimizer = torch.optim.Adam(model_lstm.parameters(), lr=0.001)


# In[80]:


def train_epoch_lstm(model, loader):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg)

        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1).to(device)

        loss = criterion(output, trg)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# In[21]:


# Train the model for multiple epochs
for epoch in range(10):
    loss = train_epoch_lstm(model_lstm, train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {loss:.4f}")


# The model is trained for 10 epochs
# 
# Each epoch means:
# 
# -the entire training dataset is seen once
# 
# The training loss is printed to:
# 
# -monitor convergence
# 
# -detect underfitting or overfitting

# In[37]:


def generate_question(model_lstm, sentence, vocab, max_len=20):
    """
    Generate a question given an input context sentence
    using the trained Seq2Seq model.
    """

    # Set model to evaluation mode
    # (disables dropout and other training-only layers)
    model_lstm.eval()
    
    # Create a reverse vocabulary mapping:
    # index -> word (needed to convert predictions back to text)
    idx_to_word = {idx: word for word, idx in vocab.items()}

     # Disable gradient computation during inference
    with torch.no_grad():
        # Encode the input context sentence into token indices
        src = encode_sentence(sentence, vocab, max_len=50)
        # Add batch dimension and move tensor to device (CPU/GPU)
        src = src.unsqueeze(0).to(device)

        # Pass input through encoder to obtain hidden and cell states
        hidden, cell = model_lstm.encoder(src)

        # Initialize decoder input with <SOS> token
        x = torch.tensor([vocab[SOSTOKEN]]).to(device)

        # List to store generated words
        generated_tokens = []

        # Generate tokens one by one (autoregressive decoding)
        for _ in range(max_len):
            # Pass previous token and hidden state to decoder
            output, hidden, cell = model_lstm.decoder(x, hidden, cell)
            # Select the word with the highest probability (greedy decoding)
            pred_idx = output.argmax(1).item()
            
            # Stop generation when <EOS> token is produced
            if idx_to_word[pred_idx] == EOSTOKEN:
                break
                
            # Append predicted word to output sequence
            generated_tokens.append(idx_to_word[pred_idx])
            # Use predicted token as next decoder input
            x = torch.tensor([pred_idx]).to(device)
    # Join generated tokens into a single string
    return " ".join(generated_tokens)


# In[81]:


sample = vali_data[0]

print("Context:")
print(sample['text'])

print("\nGold Question:")
print(sample['gold_label_str'])

print("\nGenerated Question:")
print(generate_question(model_lstm, sample['text'], vocab))


# Displays:
# 
# -the input context
# 
# -the ground-truth (gold) question
# 
# -the model-generated question
# 
# In the above output section for generated question 

# In[56]:


for i in range(5):
    print("\n----------------------------")
    print("Context:")
    print(vali_data[i]['text'])

    print("\nGold Question:")
    print(vali_data[i]['gold_label_str'])

    print("\nGenerated Question:")
    print(generate_question(model_lstm, vali_data[i]['text'], vocab))


# Prints 5 validation examples
# Allows:
# 
# -manual inspection
# 
# -comparison between gold and generated questions
# 

# # Fine tunning - T5 Model

# 
# This is the most standard and accurate approach for Question Generation.
# 
# We’ll use T5, because it is explicitly designed for text-to-text tasks.

# In[40]:


get_ipython().system('pip install transformers datasets sentencepiece accelerate')



# In[41]:


pip install ipywidgets


# In[42]:


pip install hf_xet


# Following is the reason why we are using it
# 
# transformers → Provides pretrained NLP models like T5
# 
# datasets → Helps load and process datasets efficiently
# 
# sentencepiece → Tokenization tool required by T5
# 
# accelerate → Improves training speed (especially on GPUs)
# 
# ipywidgets → Helpful for interactive notebooks
# 
# hf_xet → Optimized file handling for Hugging Face

# In[43]:


import json
from datasets import Dataset
# json → Used to read dataset files stored in JSON format
#Dataset → Converts raw data into Hugging Face’s dataset structure

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# The above function Opens a JSONL file (one JSON object per line),Reads each line,Converts it into a Python dictionary,Returns a list of data samples
train_data = load_jsonl("qg_train_cleaned.jsonl")
vali_data   = load_jsonl("qg_validation_cleaned.jsonl")

#train_data → Used to teach the model
#vali_data → Used to evaluate model performance

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
vali_dataset   = Dataset.from_list(vali_data)

'''Hugging Face models cannot directly use raw Python lists.This step converts the data into an optimized format that supports:
Mapping

Tokenization

Batching
'''


# In[44]:


from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess(batch):
# Adds instruction text "generate question:",Helps T5 understand what task to perform,This is how T5 supports multiple NLP tasks
    inputs = ["generate question: " + x for x in batch["text"]] 
#These are the correct questions, the model tries to learn to generate these
    targets = batch["gold_label_str"]


    model_inputs = tokenizer(
        inputs,
        truncation=True,     #Cuts very long text
        padding="max_length",  #Ensures same input length
        max_length=512       #Standard max size for T5 input
    )

    labels = tokenizer(
        targets,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess, batched=True)
vali_dataset   = vali_dataset.map(preprocess, batched=True)


# A tokenizer:
# 
# Converts text into numbers
# 
# Breaks sentences into meaningful tokens
# 
# Makes text understandable to neural networks

# In[45]:


from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments


# Loads a pretrained T5 model
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-small")


training_args = TrainingArguments(
    output_dir="./t5_qg",
    eval_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=100,
    report_to="none"
)


#          Parameter -	Meaning
#         output_dir -	Where trained model is saved
#     learning_rate  -	Speed of learning
#       batch_size   -    Samples processed together
#     num_train_epochs -    Number of training cycles
#       weight_decay -	Prevents overfitting
#      eval_strategy -	Evaluates after each epoch

# In[47]:


trainer = Trainer(
    model=model_t5,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=vali_dataset,
    tokenizer=tokenizer
)

trainer.train()

# Here Trainer is kind of function which handels forward pass, loss calculation, backpropagation, evaluation 
# in the above code epoch is trained and it checks the training loss and validation loss based on the data provided 
''' In this trainer.train() what exactly happens is:
It reads input text

generates a question

compares with correct question

adjusts internal weights

repeats for all data '''


# In[52]:


# This function uses the trained model to generate new questions
def generate_question_t5(model_t5, tokenizer, context):
    device = model_t5.device  # get model device (cpu or cuda)

    input_text = "generate question: " + context
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt"
    ).to(device)  # 🔑 move tensor to same device

    outputs = model_t5.generate(
        input_ids,
        max_length=64,
        num_beams=4,          #Uses beam search for better question quality and Stops when best output is found
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True) # converts model output back into readable text 

# Selects one validation example Prints: Original paragraph,Generated question
sample = vali_data[i]
print("Context:", sample["text"])
print("\n")
print("Generated:", generate_question_t5(model_t5, tokenizer, sample["text"]))
print("\n")
   


# In[53]:


for i in range(5):
    print("\n----------------------------")
    print("Context:")
    print(vali_data[i]['text'])

    print("\nGold Question:")
    print(vali_data[i]['gold_label_str'])

    print("\nGenerated Question:")
    print(generate_question_t5(model_t5,tokenizer,vali_data[i]['text']))


# # Now both models chosen is completed trained and getting the output. Now lets compare with other models and check the results of it so for comparison we are choosing Vanilla RNN and BERT respectively

# # Vanilla RNN

# A Vanilla RNN (Recurrent Neural Network) is the simplest and most basic form of a recurrent neural network. It is called “vanilla” because it has no special mechanisms to handle long-term dependencies.

# Vanilla RNN was included as a baseline model to compare classical sequence learning methods with gated recurrent networks and transformer-based models.”

# In[57]:


import torch.nn as nn

class EncoderRNN(nn.Module):
    """
    Encoder:
    - Converts input words into numbers (embeddings)
    - Reads the entire input sentence
    - Produces a hidden state that represents the input meaning
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)# Converts word IDs into dense vectors
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)# Vanilla RNN that processes the sequence

    def forward(self, x):
        embedded = self.embedding(x) # Convert input words to embeddings
        _, hidden = self.rnn(embedded)  # Pass embeddings through RNN and We only keep the final hidden state
        
        return hidden



# -Maps input tokens to embeddings
# 
# -LSTM summarizes entire sequence into final hidden and cell states

# In[58]:


class DecoderRNN(nn.Module):
    """
    Decoder:
    - Takes the encoder’s hidden state
    - Generates output words one at a time
    """
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        # Converts output word IDs into embeddings
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
         # RNN that generates one word at each time step
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
         # Converts RNN output into vocabulary probabilities
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        # Add time dimension (needed for RNN)
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)# Generate next hidden state
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


# In[82]:


class Seq2SeqRNN(nn.Module):
    """
    Combines Encoder and Decoder into a single
    Seq2Seq model.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)

        # Encode input sequence
        hidden = self.encoder(src)
        # First input to decoder is <SOS>
        x = trg[:, 0]  # <SOS>

        # Teacher forcing
        for t in range(1, trg_len):
            output, hidden = self.decoder(x, hidden)
            outputs[:, t] = output
            x = trg[:, t]  # teacher forcing
       

        return outputs


# In[60]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = EncoderRNN(vocab_size, 100, 256)#100 → size of word embeddings 256 → size of hidden memory
decoder = DecoderRNN(vocab_size, 100, 256)
model_rnn = Seq2SeqRNN(encoder, decoder).to(device)

#Loss function measures how wrong the predictions are

#Adam optimizer updates model weights to reduce errors
criterion = nn.CrossEntropyLoss(ignore_index=vocab[PADTOKEN])
optimizer = torch.optim.Adam(model_rnn.parameters(), lr=0.001)


# In[61]:


def train_epoch(model, loader):
    model.train()
    total_loss = 0

    for src, trg in loader:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model_rnn(src, trg)

        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# In[62]:


for epoch in range(10):
    loss = train_epoch(model_rnn, train_loader)
    print(f"Epoch {epoch+1} | Train Loss: {loss:.4f}")

    


# The model is trained for 5 epochs
# 
# Each epoch means:
# 
# -the entire training dataset is seen once
# 
# The training loss is printed to:
# 
# -monitor convergence
# 
# -detect underfitting or overfitting

# In[63]:


def generate_question_rnn(model, sentence, vocab, max_len=20):
    """
    Generate a question given an input context sentence
    using the trained Vanilla RNN Seq2Seq model.
    """

    model.eval()
    idx_to_word = {idx: word for word, idx in vocab.items()}

    with torch.no_grad():
        src = encode_sentence(sentence, vocab, max_len=50)
        src = src.unsqueeze(0).to(device)

        # ✅ RNN encoder returns ONLY hidden
        hidden = model.encoder(src)

        x = torch.tensor([vocab[SOSTOKEN]]).to(device)
        generated_tokens = []

        for _ in range(max_len):
            output, hidden = model.decoder(x, hidden)
            pred_idx = output.argmax(1).item()

            if idx_to_word[pred_idx] == EOSTOKEN:
                break

            generated_tokens.append(idx_to_word[pred_idx])
            x = torch.tensor([pred_idx]).to(device)

    return " ".join(generated_tokens)


# In[64]:


sample = vali_data[0]

print("Context:")
print(sample['text'])

print("\nGold Question:")
print(sample['gold_label_str'])

print("\nGenerated Question (RNN):")
print(generate_question_rnn(model_rnn, sample['text'], vocab))


# In[83]:


for i in range(5):
    print("\n----------------------------")
    print("Context:")
    print(vali_data[i]['text'])

    print("\nGold Question:")
    print(vali_data[i]['gold_label_str'])

    print("\nGenerated Question:")
    print(generate_question_rnn(model_rnn, vali_data[i]['text'], vocab))


# # BERT

# Although BERT is a powerful transformer-based model, it is fundamentally an encoder-only architecture designed for understanding tasks such as classification and extractive question answering. Automatic question generation is a sequence-to-sequence text generation task that requires autoregressive decoding, which BERT does not support natively. Implementing BERT for question generation would require architectural modifications such as adding a decoder or using specialized variants like UniLM, which is beyond the scope of this project. In contrast, T5 is an encoder–decoder transformer explicitly designed for text generation tasks and reformulates all NLP problems in a text-to-text format. Therefore, T5 was selected as the transformer-based model for this project.

# # EVALUATION MATRIX
# 

# # BLEU

# BLEU (Bilingual Evaluation Understudy): -measures n-gram overlap between generated and reference questions SmoothingFunction: -avoids zero BLEU scores for short sequences

# BLEU compares:
# 
# -generated question (hyp)
# 
# -reference question (ref)
# 
# Only first n_samples validation examples are used for efficiency
# 
# Higher BLEU = better n-gram similarity

# In[65]:


pip install nltk


# In[66]:


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# In[67]:


def compute_bleu_lstm(model, data, vocab, n_samples=100):
    smoothie = SmoothingFunction().method4
#List to store BLEU scores for each sample
    scores = []
    
    for i in range(n_samples):
        ref = data[i]['gold_label_str'].split()
# Generate a question using the LSTM model
# Split generated sentence into words
        hyp = generate_question(model, data[i]['text'], vocab).split()

        if len(hyp) == 0:
            continue

        score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
        scores.append(score)

    return sum(scores) / len(scores)


# In[68]:


bleu_lstm = compute_bleu_lstm(model_lstm, vali_data, vocab)
print(f"BLEU Score (LSTM): {bleu_lstm:.4f}")


# In[69]:


def compute_bleu_t5(model, tokenizer, data, n_samples=100):
    smoothie = SmoothingFunction().method4
    scores = []
# Put model in evaluation mode (no training)
    model.eval()

    for i in range(n_samples):
        reference = data[i]['gold_label_str'].split()
# Generate question using T5 model
        hypothesis = generate_question_t5(
            model,
            tokenizer,
            data[i]['text']
        ).split()

        if len(hypothesis) == 0:
            continue
# Compute BLEU score
        score = sentence_bleu(
            [reference],
            hypothesis,
            smoothing_function=smoothie
        )
        scores.append(score)

    return sum(scores) / len(scores)


# In[71]:


bleu_t5 = compute_bleu_t5(model_t5, tokenizer, vali_data)
print(f"BLEU Score (T5): {bleu_t5:.4f}")


# In[72]:


def compute_bleu_rnn(model, data, vocab, n_samples=100):
    smoothie = SmoothingFunction().method4
    scores = []

    model.eval()

    for i in range(n_samples):
        reference = data[i]['gold_label_str'].split()
# Generate question using RNN
        hypothesis = generate_question_rnn(
            model,
            data[i]['text'],
            vocab
        ).split()

        if len(hypothesis) == 0:
            continue
# Compute BLEU score
        score = sentence_bleu(
            [reference],
            hypothesis,
            smoothing_function=smoothie
        )
        scores.append(score)

    return sum(scores) / len(scores)


# In[73]:


bleu_rnn = compute_bleu_rnn(model_rnn, vali_data, vocab)
print(f"BLEU Score (Vanilla RNN): {bleu_rnn:.4f}")


# # ROUGE-L

# ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
# 
# commonly used in text generation and summarization
# 
# ROUGE-L:
# 
# measures longest common subsequence (LCS)

# ROUGE-L evaluates:
# 
# -sequence-level similarity
# 
# -order-aware overlap
# 
# F-measure balances precision and recall

# In[74]:


pip install rouge-score


# In[75]:


from rouge_score import rouge_scorer

#general function to evaluate sequence-to-sequence models LSTM AND RNN

def compute_rouge_seq2seq(model, data, vocab, generate_fn, n_samples=100):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 
    scores = []
#EVALUATION MODE
    model.eval()

    for i in range(min(n_samples, len(data))):
        reference = data[i]['gold_label_str']
        hypothesis = generate_fn(model, data[i]['text'], vocab)

        if hypothesis.strip() == "":
            continue

        score = scorer.score(reference, hypothesis)
        scores.append(score['rougeL'].fmeasure)

    return sum(scores) / len(scores)


# In[76]:


rouge_rnn = compute_rouge_seq2seq(
    model_rnn, vali_data, vocab, generate_question_rnn
)

rouge_lstm = compute_rouge_seq2seq(
    model_lstm, vali_data, vocab, generate_question
)

print(f"ROUGE-L (RNN): {rouge_rnn:.4f}")
print(f"ROUGE-L (LSTM): {rouge_lstm:.4f}")


# In[77]:


def compute_rouge_t5(model, tokenizer, data, n_samples=100):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

#  Evaluation mode, no training takes place
    model.eval()

    for i in range(min(n_samples, len(data))):
        reference = data[i]['gold_label_str']
# Generating t5 model
        hypothesis = generate_question_t5(
            model,
            tokenizer,
            data[i]['text']
        )

        if hypothesis.strip() == "":
            continue

        score = scorer.score(reference, hypothesis)
        scores.append(score['rougeL'].fmeasure)

    return sum(scores) / len(scores)


# In[79]:


rouge_t5 = compute_rouge_t5(model_t5, tokenizer, vali_data)

print(f"ROUGE-L (T5): {rouge_t5:.4f}")


# In[ ]:




