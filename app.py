import os
import json
import re
import csv
from datetime import datetime
from typing import Tuple, Optional

import gradio as gr
from dotenv import load_dotenv

try:
    from groq import Groq
except Exception as e:  # pragma: no cover
    Groq = None  # type: ignore


CSV_PATH = os.path.join(os.path.dirname(__file__), "/result/call_analysis.csv")


def _ensure_csv_header(path: str) -> None:
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Transcript", "Summary", "Sentiment"])


def _normalize_sentiment(value: str) -> str:
    v = (value or "").strip().lower()
    # Map common variants to {positive, neutral, negative}
    mapping = {
        "pos": "positive",
        "+": "positive",
        "positive": "positive",
        "happy": "positive",
        "satisfied": "positive",
        "delighted": "positive",
        "great": "positive",
        "ok": "neutral",
        "meh": "neutral",
        "neutral": "neutral",
        "mixed": "neutral",
        "neg": "negative",
        "-": "negative",
        "negative": "negative",
        "frustrated": "negative",
        "angry": "negative",
        "upset": "negative",
    }
    return mapping.get(v, "negative" if any(w in v for w in ["frustrat", "angry", "upset", "mad", "disappoint"]) else ("positive" if any(w in v for w in ["happy", "satisf", "delight", "great"]) else "neutral"))


def _extract_json_obj(text: str) -> Optional[dict]:
    """Try to extract the first JSON object from the model text response."""
    if not text:
        return None
    # Look for the first {...} block
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def analyze_transcript(transcript: str) -> Tuple[str, str, str]:
    """
    Contract
    - Input: transcript (str), non-empty customer call/dialog text
    - Output: (summary, sentiment, status)
      - summary: 2–3 sentence summary from Groq
      - sentiment: one of {positive, neutral, negative}
      - status: info/error message for UI
    Side effects:
      - Prints transcript/summary/sentiment to console
      - Appends a row to call_analysis.csv with headers
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return "", "", "Please enter a transcript."

    load_dotenv()  # load .env if present
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "", "", "Missing GROQ_API_KEY in environment. See README to configure .env."

    if Groq is None:
        return "", "", "Groq SDK not available. Ensure dependencies are installed."

    client = Groq(api_key=api_key)
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    system_prompt = (
        "You are an assistant that summarizes customer service calls and classifies sentiment.\n"
        "Return a strict JSON object with exactly these keys: summary, sentiment.\n"
        "- summary: 2–3 sentences, concise and factual.\n"
        "- sentiment: one of [positive, neutral, negative] only."
    )

    user_prompt = (
        "Analyze the following customer call transcript.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Respond ONLY with a JSON object like:\n"
        "{\n  \"summary\": \"...\",\n  \"sentiment\": \"positive|neutral|negative\"\n}"
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=256,
            top_p=0.9,
        )
        content = completion.choices[0].message.content if completion and completion.choices else ""
    except Exception as e:  # pragma: no cover
        return "", "", f"Groq API error: {e}"

    data = _extract_json_obj(content) or {}
    summary = (data.get("summary") or "").strip()
    sentiment_raw = (data.get("sentiment") or "").strip()
    sentiment = _normalize_sentiment(sentiment_raw)

    # Fallback summarization if model returned empty (very unlikely)
    if not summary:
        summary = "Automated: Customer discussed an issue; details shortened due to parsing."

    # Print to console as requested
    print("----- Call Analysis -----")
    print(f"Transcript: {transcript}")
    print(f"Summary:    {summary}")
    print(f"Sentiment:  {sentiment}")
    print("-------------------------")

    # Persist to CSV
    try:
        _ensure_csv_header(CSV_PATH)
        with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([transcript, summary, sentiment])
        status = f"Saved to {os.path.basename(CSV_PATH)}"
    except Exception as e:  # pragma: no cover
        status = f"Failed to save CSV: {e}"

    return summary, sentiment, status


EXAMPLE_TRANSCRIPT = (
    "Customer: Hi, I was trying to book a slot yesterday but the payment failed.\n"
    "Agent: I'm sorry to hear that. Did you receive any error message?\n"
    "Customer: It said 'transaction declined'. I tried twice and got charged once without a booking.\n"
    "Agent: I'll look into the charge and help you rebook without extra fees."
)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Call Summarizer & Sentiment (Groq)") as demo:
        gr.Markdown(
            """
            # Call Summarizer & Sentiment
            Enter a customer call transcript. The app will use the Groq API to summarize the conversation (2–3 sentences) and classify sentiment.
            Results are appended to `call_analysis.csv` in this folder.
            """
        )

        with gr.Row():
            transcript_in = gr.Textbox(label="Transcript", lines=10, value=EXAMPLE_TRANSCRIPT)

        with gr.Row():
            btn = gr.Button("Analyze", variant="primary")

        with gr.Row():
            summary_out = gr.Textbox(label="Summary", lines=4)
            sentiment_out = gr.Textbox(label="Sentiment", lines=1)

        status_out = gr.Markdown()

        def _on_click(t: str):
            s, sen, status = analyze_transcript(t)
            return s, sen, f"Status: {status}"

        btn.click(_on_click, inputs=[transcript_in], outputs=[summary_out, sentiment_out, status_out])

    return demo


if __name__ == "__main__":
    # Launch Gradio app
    app = build_interface()
    app.launch(share=True)
