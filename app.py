import re

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = FastAPI(
    title="Chat Summarizer App",
    description="Chat Summarization using T5",
    version="1.0",
)

# Load model + tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
model.eval()

# UI (index.html is in the project root)
templates = Jinja2Templates(directory=".")


class DialogueInput(BaseModel):
    dialogue: str


def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)    # line breaks
    text = re.sub(r"\s+", " ", text)     # extra spaces
    text = re.sub(r"<.*?>", " ", text)   # html tags
    return text.strip().lower()          # keep: model was trained on lowercase


def summarize_dialogue(dialogue: str) -> str:
    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            min_length=25,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
            early_stopping=True,
        )

    summary = tokenizer.decode(targets[0], skip_special_tokens=True)
    summary = re.sub(r"^[\s.,;:\-]+", "", summary)   # strip leading junk
    return summary[:1].upper() + summary[1:]          # fix casing


# Plain def -> runs in threadpool, server doesn't freeze during generate()
@app.post("/summarize/")
def summarize(dialogue_input: DialogueInput):
    if not dialogue_input.dialogue.strip():
        return {"summary": "No text provided."}
    return {"summary": summarize_dialogue(dialogue_input.dialogue)}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")