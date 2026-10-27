#  Chat Summarizer using HuggingFace T5 & FastAPI

An end-to-end, transformer-based chat summarization web application built with **FastAPI**, **HuggingFace Transformers (T5)**, and a **modern Glassmorphism UI**.

---

##  Features

- **Transformer-Powered Summarization:** Fine-tuned T5 conditional generation model capable of synthesizing conversations, dialogues, and paragraphs.
- **Beam Search Decoding:** Utilizes 5 beams, length penalties, and n-gram repetition blocking for natural, coherent summaries.
- **FastAPI Backend:** Lightweight, asynchronous Python REST API with Jinja2 templating.
- **Modern Glassmorphic UI:**
  - Dynamic, shifting multi-color gradient background (pure CSS)
  - Frosted glass container with responsive typography (Inter)
  - Interactive shimmer hover states & smooth micro-animations
- **Git LFS Tracked:** Pre-trained model weights (`model.safetensors`, ~242 MB) version-controlled seamlessly with Git Large File Storage.

---

##  Project Structure

```text
TextSummarizer/
├── app.py                      # FastAPI server & inference pipeline
├── index.html                  # Frontend web page (HTML5 + Glassmorphism CSS + JS)
├── saved_summary_model/        # Model weights & tokenizer files (tracked via Git LFS)
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors       # T5 model weights (~242 MB)
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── .gitattributes              # Git LFS configuration
├── .gitignore                  # Ignored files (pycache, env, etc.)
└── README.md                   # Project documentation
```

---

##  Getting Started

### 1. Prerequisites

- Python 3.9+
- Git & [Git LFS](https://git-lfs.com/) (required to clone model weights)

### 2. Clone the Repository

```bash
git lfs install
git clone https://github.com/Madhura-Kedar/chat-summarizer-t5-fastapi.git
cd chat-summarizer-t5-fastapi
git lfs pull
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn torch transformers pydantic jinja2
```

*(If you have CUDA or Apple Silicon MPS support, PyTorch will automatically utilize your GPU for accelerated generation).*

### 4. Run the Application

```bash
uvicorn app:app --reload --port 8000
```

Once running, open your browser and navigate to:
```
http://localhost:8000
```

---

##  API Reference

### `POST /summarize/`

Takes raw dialogue/text input and returns an abstractive summary.

**Request Body:**
```json
{
  "dialogue": "John: Hey, are we still meeting today at 5? Mary: Yes, at the coffee shop downtown. John: Great, see you there!"
}
```

**Response:**
```json
{
  "summary": "John and Mary agree to meet at the coffee shop downtown today at 5."
}
```



