"""
Job Application Assistant - powered by Llama 3.2 via Ollama (fully local)

Setup:
    ollama pull llama3.2
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import json
import re
import ollama
import pdfplumber
import streamlit as st


# ---------- helpers ----------
def read_file(upload):
    """Read text from an uploaded PDF or TXT file. Returns "" if nothing could be read."""
    if upload is None:
        return ""
    data = upload.getvalue()
    if upload.name.lower().endswith(".pdf"):
        text = ""
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as e:
            st.error(f"Could not open this PDF: {e}")
        return text.strip()
    return data.decode("utf-8", errors="ignore").strip()


def llm(prompt, model, as_json=False, temperature=0.3, retries=2):
    """Call Ollama. If as_json, parse the reply and retry on broken JSON."""
    kwargs = {"format": "json"} if as_json else {}
    for attempt in range(retries + 1):
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_ctx": 8192},
            **kwargs,
        )
        text = resp["message"]["content"]
        if not as_json:
            return text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    return {}


# ---------- pipeline steps ----------
def parse_resume(text, model):
    prompt = f"""Extract information from this resume. Reply with JSON only, in this format:
{{"name": "", "skills": [], "education": [], "bullets": []}}
"bullets" = every experience/project bullet point, copied as written.

Resume:
{text}"""
    return llm(prompt, model, as_json=True, temperature=0.1)


def parse_job(text, model):
    prompt = f"""Extract information from this job posting. Reply with JSON only, in this format:
{{"title": "", "company": "", "required_skills": [], "nice_to_have": [], "responsibilities": []}}

IMPORTANT for "required_skills": list 8-15 SHORT keyword phrases (1-3 words each) naming the
skills, tools, methods, and concepts the job needs. If the posting only lists responsibilities,
infer the skills from them (e.g. "Perform molecular docking" -> "molecular docking").

Posting:
{text}"""
    return llm(prompt, model, as_json=True, temperature=0.1)


STOPWORDS = {"and", "or", "the", "of", "in", "to", "for", "with", "a", "an", "on", "using",
             "use", "experience", "knowledge", "skills", "skill", "ability", "strong", "good",
             "understanding", "as", "by", "at", "is", "are"}


def _stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokens(text):
    words = re.findall(r"[a-z0-9+#.]+", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS and len(w) > 1}


def match_analysis(resume_text, job):
    """Forgiving keyword match: a skill counts if its phrase appears, or 60%+ of its words do."""
    skills = [s for s in job.get("required_skills", []) if isinstance(s, str) and s.strip()]
    if not skills:  # fallback if the model returned no skills
        skills = [s for key in ("nice_to_have", "responsibilities")
                  for s in job.get(key, []) if isinstance(s, str) and s.strip()]
    resume_tokens = _tokens(resume_text)
    low = resume_text.lower()
    matched, missing = [], []
    for s in dict.fromkeys(skills):
        toks = _tokens(s)
        hit = s.lower() in low or (toks and len(toks & resume_tokens) / len(toks) >= 0.6)
        (matched if hit else missing).append(s)
    total = len(matched) + len(missing)
    score = round(100 * len(matched) / total) if total else 0
    return {"score": score, "matched": matched, "missing": missing}


def tailor_bullet(bullet, job, model):
    prompt = f"""Rewrite this resume bullet to better fit the job below.
RULES:
- Use ONLY facts in the original bullet. Do NOT invent tools, numbers, or achievements.
- Start with a strong action verb and keep it to one line.
- Naturally use relevant keywords from the job if they truly apply.

Job title: {job.get('title', '')}
Key skills: {', '.join(map(str, job.get('required_skills', [])))}

Original bullet: {bullet}

Reply with JSON only: {{"rewritten": ""}}"""
    return llm(prompt, model, as_json=True).get("rewritten", bullet)


def verify_bullet(original, rewritten, model):
    """Anti-hallucination check: is the rewritten bullet supported by the original?"""
    prompt = f"""Compare an original resume bullet with a rewritten version.
Does the rewritten version add any skill, tool, number, or claim NOT present in the original?

Original: {original}
Rewritten: {rewritten}

Reply with JSON only: {{"supported": true or false, "reason": "short explanation"}}"""
    result = llm(prompt, model, as_json=True, temperature=0.0)
    return bool(result.get("supported", False)), result.get("reason", "")


def cover_letter(resume, job, tone, model):
    prompt = f"""Write a {tone} cover letter (3-4 short paragraphs, under 250 words).
Use ONLY facts from the resume. Do not invent experience.

Candidate: {resume.get('name', '')}
Skills: {', '.join(map(str, resume.get('skills', [])))}
Experience highlights: {' | '.join(map(str, resume.get('bullets', [])[:6]))}

Role: {job.get('title', '')} at {job.get('company', 'the company')}
Requirements: {', '.join(map(str, job.get('required_skills', [])))}
Responsibilities: {', '.join(map(str, job.get('responsibilities', [])[:5]))}

Structure: opening (role + interest), middle (2-3 relevant strengths tied to requirements), closing (ask for interview)."""
    return llm(prompt, model, temperature=0.6)


def interview_questions(resume, job, model):
    prompt = f"""Create interview preparation for this candidate and role.

Role: {job.get('title', '')}
Requirements: {', '.join(map(str, job.get('required_skills', [])))}
Candidate skills: {', '.join(map(str, resume.get('skills', [])))}
Candidate experience: {' | '.join(map(str, resume.get('bullets', [])[:6]))}

Give:
1. 5 technical questions likely to be asked
2. 4 behavioral questions
3. For 2 of the behavioral questions, a STAR (Situation, Task, Action, Result) answer outline
   using ONLY the candidate's real experience above."""
    return llm(prompt, model, temperature=0.5)


# ---------- UI ----------
st.set_page_config(page_title="Job Application Assistant", layout="wide")
st.title("Job Application Assistant")
st.caption("Runs 100% locally with Ollama. Your resume never leaves your laptop.")

with st.sidebar:
    model = st.selectbox("Model", ["llama3.2", "llama3.2:1b"])
    tone = st.selectbox("Cover letter tone", ["formal", "friendly", "concise"])
    max_bullets = st.slider("Max bullets to tailor", 3, 12, 6)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Your resume")
    resume_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    resume_text = read_file(resume_file)
    if resume_file and resume_text:
        st.success(f"Read {len(resume_text)} characters from {resume_file.name}")
        with st.expander("Preview extracted text"):
            st.text(resume_text[:1500])
    elif resume_file:
        st.warning("No text could be read from this PDF. It is probably a scanned image or a "
                   "design-tool export (e.g. Canva). Paste the text in the box below instead, "
                   "or re-save the PDF from Word / Google Docs.")
    if not resume_text:
        resume_text = st.text_area("...or paste your resume here", height=250)
with col2:
    st.subheader("Job posting")
    job_text = st.text_area("Paste the job description", height=330)

if st.button("Generate application kit", type="primary"):
    if not resume_text.strip() or not job_text.strip():
        st.warning("Please provide both a resume and a job posting.")
        st.stop()

    with st.spinner("Parsing resume and job posting..."):
        resume = parse_resume(resume_text, model)
        job = parse_job(job_text, model)

    tab1, tab2, tab3, tab4 = st.tabs(["Match", "Tailored bullets", "Cover letter", "Interview prep"])

    with tab1:
        m = match_analysis(resume_text, job)
        if not job.get("required_skills"):
            st.warning("The model found no skills in the job posting. Try pasting more of the "
                       "description, or switch to the llama3.2 (3B) model.")
        st.metric("Skill match", f"{m['score']}%")
        st.write("**Matched:**", ", ".join(m["matched"]) or "none")
        st.write("**Missing (consider learning or adding if true):**", ", ".join(m["missing"]) or "none")
        with st.expander("Parsed data (debug)"):
            st.json({"resume": resume, "job": job})

    with tab2:
        bullets = [b for b in resume.get("bullets", []) if isinstance(b, str)][:max_bullets]
        if not bullets:
            st.info("No bullets found in the resume.")
        progress = st.progress(0.0)
        for i, original in enumerate(bullets):
            new = tailor_bullet(original, job, model)
            ok, reason = verify_bullet(original, new, model)
            st.markdown(f"**Original:** {original}")
            st.markdown(f"**Tailored:** {new}")
            if ok:
                st.success("Verified: no invented claims")
            else:
                st.error(f"Possible hallucination: {reason}")
            st.divider()
            progress.progress((i + 1) / len(bullets))

    with tab3:
        with st.spinner("Writing cover letter..."):
            st.write(cover_letter(resume, job, tone, model))

    with tab4:
        with st.spinner("Preparing interview questions..."):
            st.write(interview_questions(resume, job, model))
