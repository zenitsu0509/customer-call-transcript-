# Call Summarizer & Sentiment (Groq + Gradio)

This small app takes a customer call transcript, uses the Groq API to:
- Summarize the conversation in 2–3 sentences.
- Extract sentiment: positive / neutral / negative.

It prints the transcript, summary, and sentiment to the console, and appends the results to `call_analysis.csv`.

## Quickstart

Prereqs:
- Python 3.9+ installed.
- A Groq API key from https://console.groq.com. Set it in an `.env` file.

### 1) Clone / open this folder

### 2) Create and populate `.env`

Create `.env` next to `app.py` with:

```
GROQ_API_KEY=your_groq_api_key_here
# Optional: override model (defaults to llama3-8b-8192)
# GROQ_MODEL=llama-3.1-8b-instant
```

### 3) Install deps

On Windows (bash):

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

If `source` isn't available, use:

```bash
. .venv/Scripts/activate
```

### 4) Run the app

```bash
python app.py
```

Gradio will print a local URL (e.g., http://127.0.0.1:7860). Open it in your browser.

### 5) Try it
- Keep the default sample transcript or paste your own.
- Click "Analyze".
- Check the console for printed results.
- The CSV `call_analysis.csv` will be created/appended with columns: Transcript | Summary | Sentiment.

## File Overview
- `app.py` — Gradio UI, Groq API call, CSV persistence.
- `requirements.txt` — Python dependencies.
- `.env` — Local environment variables (not committed). See `.env.example`.

## Suggested sentiment models (local alternatives)
If you prefer a local classifier instead of calling Groq for sentiment, here are lightweight options you can integrate:

1. VADER (NLTK)
   - Pros: Very fast, no GPU, great for social/customer text.
   - Install: `pip install nltk` then in Python: `nltk.download('vader_lexicon')`
   - Usage:
     ```python
     from nltk.sentiment import SentimentIntensityAnalyzer
     sia = SentimentIntensityAnalyzer()
     score = sia.polarity_scores(text)['compound']
     sentiment = 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'
     ```

2. Hugging Face Transformers small pipelines
   - `pip install transformers torch --upgrade`
   - Example model: `distilbert-base-uncased-finetuned-sst-2-english`
   - Usage:
     ```python
     from transformers import pipeline
     clf = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
     label = clf(text)[0]['label'].lower()  # 'positive' or 'negative'
     sentiment = 'neutral' if label not in ('positive', 'negative') else label
     ```

3. Tiny Roberta variant (3-way sentiment)
   - `cardiffnlp/twitter-roberta-base-sentiment`
   - Gives 3 classes: negative/neutral/positive

In this app, Groq already returns `summary` and `sentiment`. To switch to a local classifier, replace the `_normalize_sentiment`/Groq sentiment with a call to one of the models above and keep CSV output the same.


## Upgrading Sentiment Analysis (better accuracy)

If you need higher-quality sentiment classification (domain-specific phrases, multi-turn nuance), consider these options:

### Option 1: Strong off-the-shelf transformers

- English, 2-class (pos/neg): `distilbert-base-uncased-finetuned-sst-2-english` (fast, reliable)
- English, 3-class (neg/neu/pos): `cardiffnlp/twitter-roberta-base-sentiment` (balanced for social/customer tone)
- Multilingual: `nlptown/bert-base-multilingual-uncased-sentiment` (1–5 stars → map to neg/neu/pos)

Example usage:

```python
from transformers import pipeline

# 3-class output (negative/neutral/positive)
clf = pipeline(
            'sentiment-analysis',
            model='cardiffnlp/twitter-roberta-base-sentiment',
            return_all_scores=False
)
label = clf(text)[0]['label'].lower()  # e.g., 'negative', 'neutral', 'positive'
```

### Option 2: Domain adaptation via fine-tuning

- Goal: adapt the model to your transcripts (e.g., booking flows, refund wording, app-specific jargon).
- Options:
      - Full fine-tuning: best accuracy; requires more compute and data.
      - Parameter-efficient fine-tuning (PEFT/LoRA): lightweight, good results with small GPUs.
- Minimal data: start with ~500–2000 labeled utterances/snippets, balanced across classes; label at the utterance or turn level.

Sketch with PEFT + Trainer (binary/3-class):

```python
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_name = 'distilbert-base-uncased'
num_labels = 3  # negative/neutral/positive

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# LoRA wrap (reduce memory/compute)
peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_lin','k_lin','v_lin','out_lin'], lora_dropout=0.05, bias='none', task_type='SEQ_CLS')
model = get_peft_model(base_model, peft_config)

# Prepare dataset
data = [
            {'text': 'payment failed and no one helped', 'label': 0},  # negative
            {'text': 'issue resolved quickly, thanks!', 'label': 2},    # positive
            {'text': 'just checking schedule', 'label': 1},             # neutral
]
ds = Dataset.from_list(data)

def tokenize(batch):
            return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

ds = ds.map(tokenize, batched=True)
ds = ds.class_encode_column('label')
ds = ds.train_test_split(test_size=0.2, seed=42)

args = TrainingArguments(
            output_dir='./sentiment_model',
            learning_rate=3e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=3,
            evaluation_strategy='epoch',
            fp16=False
)

trainer = Trainer(model=model, args=args, train_dataset=ds['train'], eval_dataset=ds['test'])
trainer.train()
trainer.save_model('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')
```

Inference with your fine-tuned model:

```python
from transformers import pipeline
clf = pipeline('sentiment-analysis', model='./sentiment_model')
label = clf(text)[0]['label'].lower()
```

### Option 3: Evaluation tips

- Use a small labeled validation set (e.g., 200–500 examples) reflecting real calls.
- Metrics: accuracy, macro-F1 (important for class balance), and confusion matrix.
- Edge cases: sarcasm, mixed sentiment in long calls, negation ("not bad"), and multi-issue dialogs.

### Option 4: Integration into this app

- Replace the Groq-provided sentiment with local prediction:
      - Load a pipeline/model at app start.
      - After summary from Groq, compute sentiment locally and map to {positive, neutral, negative}.
- Keep CSV format unchanged: `Transcript,Summary,Sentiment`.
- Optionally add a dropdown in the UI to choose `Groq` vs `Local` sentiment.

## Recording guidance (4–5 minutes)

- Explain approach: simple Gradio UI; one click calls Groq; JSON parsed; results printed and saved.
- Walk through `app.py` briefly: input → `analyze_transcript()` → Groq → print → CSV.
- Show run: start app, paste transcript, analyze.
- Open `call_analysis.csv` to show saved row(s).

## Troubleshooting

- `Missing GROQ_API_KEY` — ensure `.env` exists and key is valid.
- Corporate proxies/SSL — set `REQUESTS_CA_BUNDLE` if needed.
- Port in use — Gradio picks a new one automatically; watch terminal output.
