# ResumeForge

A job application assistant that runs **100% locally** using Llama 3.2 through Ollama. Your resume never leaves your computer.

Give it a resume and a job posting, and it produces:

- **Match analysis:** a skill-match score with matched and missing skills
- **Tailored resume bullets:** rewritten to fit the job using only facts from your original
- **Anti-hallucination check:** a second model pass flags any rewritten bullet that adds claims not in the original
- **Cover letter:** in a formal, friendly, or concise tone
- **Interview prep:** technical and behavioral questions with STAR answer outlines

## How it works

```
Resume ──► Parser ──► Resume JSON ─┐
                                   ├─► Match score ─► Tailored bullets ─► Verifier
Job posting ─► Parser ─► Job JSON ─┘                ├─► Cover letter
                                                    └─► Interview questions
```

Each stage is a separate, focused prompt. Output is requested as JSON and retried if it is malformed.

This app runs locally (it uses Ollama), so there is no public website.
After running `streamlit run app.py`, open: http://localhost:8501


## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- About 8 GB of RAM (a GPU is optional but makes it faster)

## Setup

```bash
# 1. Download the model (one time, about 2 GB)
ollama pull llama3.2

# 2. Clone the project
git clone https://github.com/YOUR-USERNAME/resumeforge.git
cd resumeforge

# 3. Create a virtual environment and install dependencies
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501.

## Usage

1. Upload a resume (PDF or TXT) or paste its text.
2. Paste the job description.
3. Pick a model and cover-letter tone in the sidebar.
4. Click **Generate application kit**.

## Tech stack

Python, Streamlit, Ollama (Llama 3.2), pdfplumber

## Limitations

- Requires Ollama running locally, so it cannot be hosted on free cloud platforms.
- Scanned (image-only) PDFs are not supported; paste the text instead.
- The match score is keyword-based and may miss synonyms.
- Small local models can make mistakes, so always review the output before sending an application.

## Screenshots

_Add screenshots of the app here._

## License

MIT
