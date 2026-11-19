"""
Radiology Report Generator - prototype (research/demo only, NOT for clinical use).

Backends:
  BACKEND=hf      -> local open VLM via transformers (default: google/medgemma-4b-it)
  BACKEND=claude  -> Claude API (needs ANTHROPIC_API_KEY)

  pip install gradio transformers accelerate torch pillow anthropic
  python radiology_report_app.py
  MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct python radiology_report_app.py
  BACKEND=claude python radiology_report_app.py
"""
import base64
import html
import io
import os
import re
import time

import gradio as gr

BACKEND = os.getenv("BACKEND", "hf")
MODEL_ID = os.getenv("MODEL_ID", "google/medgemma-4b-it")  # gated: accept license on HF + `huggingface-cli login`
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MODEL_LABEL = MODEL_ID.split("/")[-1] if BACKEND == "hf" else CLAUDE_MODEL

SYSTEM = "You are an expert radiologist writing a structured report. Be precise and do not invent findings."

PROMPT = """Write a radiology report for this {modality} image.
Clinical context: {clinical}

Format:
TECHNIQUE:
FINDINGS: (region by region)
IMPRESSION: (numbered, most important first)

If image quality is limited or a finding is uncertain, say so explicitly."""

# ---------------- Backends ----------------
if BACKEND == "hf":
    import torch
    from transformers import pipeline

    pipe = pipeline("image-text-to-text", model=MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")

    def run(image, prompt, max_tokens):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ]},
        ]
        out = pipe(text=messages, max_new_tokens=max_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]
else:
    import anthropic

    client = anthropic.Anthropic()

    def run(image, prompt, max_tokens):
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        return msg.content[0].text


# ---------------- Report rendering ----------------
SECTIONS = ["Technique", "Findings", "Impression"]


def split_sections(text):
    """Split model output into Technique / Findings / Impression (tolerates markdown bold)."""
    clean = text.replace("**", "").replace("##", "").strip()
    pattern = r"(?im)^\s*(technique|findings?|impression|conclusion)s?\s*:?\s*"
    parts = re.split(pattern, clean)
    if len(parts) < 3:
        return None
    found = {}
    for i in range(1, len(parts) - 1, 2):
        key = parts[i].lower()
        key = "Findings" if key.startswith("finding") else "Impression" if key in ("impression", "conclusion") else "Technique"
        found[key] = parts[i + 1].strip()
    return found


def body_to_html(body):
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    out, in_list = [], False
    for l in lines:
        m = re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)", l)
        if m:
            if not in_list:
                out.append("<ol>" if re.match(r"^\d", l) else "<ul>")
                in_list = "ol" if re.match(r"^\d", l) else "ul"
            out.append(f"<li>{html.escape(m.group(1))}</li>")
        else:
            if in_list:
                out.append(f"</{in_list}>")
                in_list = False
            out.append(f"<p>{html.escape(l)}</p>")
    if in_list:
        out.append(f"</{in_list}>")
    return "".join(out)


def render_report(text, seconds):
    sections = split_sections(text)
    if sections:
        blocks = "".join(
            f'<section class="{s.lower()}"><h3>{s}</h3>{body_to_html(sections[s])}</section>'
            for s in SECTIONS if s in sections
        )
    else:
        blocks = f"<section>{body_to_html(text)}</section>"
    meta = f'<div class="meta">{html.escape(MODEL_LABEL)} · generated in {seconds:.1f} s · draft, not reviewed</div>'
    return f'<div class="paper">{blocks}{meta}</div>'


EMPTY = '<div class="paper empty"><p>The draft report will appear here.</p></div>'
LOADING = '<div class="paper empty"><div class="spinner"></div><p>Reading the image and drafting the report…</p></div>'


def generate(image, modality, clinical, max_tokens):
    if image is None:
        yield '<div class="paper empty"><p>Upload an image to get started.</p></div>', ""
        return
    yield LOADING, ""
    prompt = PROMPT.format(modality=modality, clinical=(clinical or "").strip() or "Not provided")
    t0 = time.time()
    try:
        text = run(image, prompt, int(max_tokens))
    except Exception as e:  # show a readable error instead of a stack trace
        yield f'<div class="paper error"><p><b>Generation failed.</b> {html.escape(str(e))}</p></div>', ""
        return
    yield render_report(text, time.time() - t0), text


# ---------------- UI ----------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
#header h1 { font-size: 1.55rem; font-weight: 600; margin: 0; }
#header p { margin: .3rem 0 0; opacity: .7; }
#notice { background: #3a2f17; border: 1px solid #6b5522; color: #f1dfb0; border-radius: 8px;
          padding: 10px 14px; font-size: .88rem; margin: 12px 0 18px; }
#viewer { background: #080b0e; border-radius: 10px; border: 1px solid #1f2a33; }
#viewer img { object-fit: contain; }
#generate { font-weight: 600; }
.paper { background: #fbfaf7; color: #1d2329; border: 1px solid #d9d6cf; border-radius: 10px;
         padding: 26px 32px; min-height: 440px; font-family: 'IBM Plex Serif', Georgia, serif;
         line-height: 1.65; font-size: .98rem; }
.paper p { margin: 0 0 .6rem; max-width: 70ch; color: #1d2329; }
.paper h3 { font-family: 'IBM Plex Sans', sans-serif; font-size: .95rem; font-weight: 600;
            color: #24566f; margin: 1.5rem 0 .4rem; }
.paper section:first-child h3 { margin-top: 0; }
.paper ol, .paper ul { margin: 0 0 .6rem 1.2rem; padding: 0; max-width: 68ch; }
.paper li { margin-bottom: .35rem; color: #1d2329; }
.paper .impression { border-left: 3px solid #24566f; padding-left: 16px; margin-top: 1.5rem; }
.paper .meta { font-family: 'IBM Plex Sans', sans-serif; font-size: .78rem; color: #7a7f85;
               margin-top: 1.8rem; padding-top: .8rem; border-top: 1px solid #e6e3dc; }
.paper.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; }
.paper.empty p { color: #7a7f85; font-family: 'IBM Plex Sans', sans-serif; }
.paper.error p { color: #9b2c2c; }
.spinner { width: 26px; height: 26px; border: 3px solid #d9d6cf; border-top-color: #24566f;
           border-radius: 50%; animation: spin .9s linear infinite; margin-bottom: 12px; }
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
"""

theme = gr.themes.Base(
    primary_hue="sky",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "sans-serif"],
    radius_size="md",
)

with gr.Blocks(theme=theme, css=CSS, title="Radiology report drafting") as demo:
    gr.HTML(
        f'<div id="header"><h1>Radiology report drafting</h1>'
        f"<p>Upload an image, add clinical context, and review the generated draft. Model: {html.escape(MODEL_LABEL)}</p></div>"
    )
    gr.HTML('<div id="notice">Research prototype. A qualified radiologist must review every finding before it is used.</div>')

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image = gr.Image(type="pil", label="Image", elem_id="viewer", height=420)
            modality = gr.Dropdown(
                ["Chest X-ray", "CT slice", "MRI slice", "Other"], value="Chest X-ray", label="Modality"
            )
            clinical = gr.Textbox(
                label="Clinical context (optional)",
                placeholder="e.g. 65-year-old male, cough and fever for 5 days",
                lines=2,
            )
            with gr.Accordion("Settings", open=False):
                max_tokens = gr.Slider(128, 1024, value=512, step=64, label="Max new tokens")
            with gr.Row():
                clear = gr.Button("Clear")
                go = gr.Button("Generate report", variant="primary", elem_id="generate")

        with gr.Column(scale=6):
            report = gr.HTML(EMPTY, label="Report")
            with gr.Accordion("Plain text (copy)", open=False):
                plain = gr.Textbox(show_label=False, lines=10, show_copy_button=True, interactive=False)

    go.click(generate, [image, modality, clinical, max_tokens], [report, plain])
    clear.click(lambda: (None, "", EMPTY, ""), None, [image, clinical, report, plain])

if __name__ == "__main__":
    demo.launch()