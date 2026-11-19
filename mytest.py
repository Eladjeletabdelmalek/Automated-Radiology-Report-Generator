"""
Radiology report drafting - NiceGUI + Ollama (local vision model).

  pip install nicegui ollama
  ollama pull gemma3:4b
  python nicegui_app.py          # opens http://localhost:8080
"""
import base64
import json
import os
import time

from nicegui import ui
from ollama import AsyncClient, Client

HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
if not HOST.startswith("http"):
    HOST = "http://" + HOST

SYSTEM = "You are an expert radiologist writing a structured report. Be precise and do not invent findings."
PROMPT = """Write a radiology report for this {modality} image.
Clinical context: {clinical}

Format:
**TECHNIQUE:**
**FINDINGS:** (region by region)
**IMPRESSION:** (numbered, most important first)

If image quality is limited or a finding is uncertain, say so explicitly."""
MODALITIES = ["Chest X-ray", "CT slice", "MRI slice", "Other"]


def vision_models():
    """Installed Ollama models that can read images (broken/missing-blob models are skipped)."""
    client = Client(host=HOST)
    try:
        res = client.list()
    except Exception:
        return []
    items = res["models"] if isinstance(res, dict) else res.models
    names = []
    for m in items:
        n = (m.get("model") or m.get("name")) if isinstance(m, dict) else (getattr(m, "model", None) or getattr(m, "name", None))
        if not n:
            continue
        try:
            info = client.show(n)
            caps = info["capabilities"] if isinstance(info, dict) else getattr(info, "capabilities", None)
        except Exception:
            continue
        if caps is None or "vision" in caps:
            names.append(n)
    return names


CSS = """
body { font-family: 'IBM Plex Sans', sans-serif; background: #0d1217; }
.viewer { background: #080b0e; border: 1px solid #1f2a33; border-radius: 12px; height: 380px; overflow: hidden; }
.notice { background: #3a2f17; border: 1px solid #6b5522; color: #f1dfb0; border-radius: 8px;
          padding: 10px 14px; font-size: .88rem; }
.paper { background: #fbfaf7; border: 1px solid #d9d6cf; border-radius: 12px; padding: 28px 34px; min-height: 520px;
         font-family: 'IBM Plex Serif', Georgia, serif; line-height: 1.65; }
.paper, .paper * { color: #1d2329; }
.paper .nicegui-markdown p { margin: 0 0 .6rem; max-width: 70ch; }
.paper .nicegui-markdown strong { font-family: 'IBM Plex Sans', sans-serif; color: #24566f; font-weight: 600; }
.paper .nicegui-markdown ol, .paper .nicegui-markdown ul { margin: 0 0 .6rem 1.2rem; }
.paper .nicegui-markdown h1, .paper .nicegui-markdown h2, .paper .nicegui-markdown h3 {
    font-family: 'IBM Plex Sans', sans-serif; font-size: 1rem; font-weight: 600; color: #24566f; margin: 1.2rem 0 .4rem; }
.paper .meta { font-family: 'IBM Plex Sans', sans-serif; font-size: .78rem; color: #7a7f85 !important; }
.paper .placeholder { color: #7a7f85 !important; font-family: 'IBM Plex Sans', sans-serif; }
"""

@ui.page("/")
def index():
    ui.add_head_html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500&'
        'family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">'
        f"<style>{CSS}</style>"
    )
    ui.colors(primary="#4FA3C7")
    ui.dark_mode().enable()
    state = {"image": None, "text": ""}
    models = vision_models()

    async def on_upload(e):
        if hasattr(e, "file"):  # NiceGUI 3.x
            data = await e.file.read()
        else:  # NiceGUI 2.x
            data = e.content.read()
        state["image"] = data
        mime = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
        url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        viewer.clear()
        with viewer:
            ui.image(url).props("fit=contain").classes("w-full h-full")
        upload.reset()

    async def generate():
        if state["image"] is None:
            ui.notify("Upload an image first", type="warning")
            return
        if not model.value:
            ui.notify("No vision model available. Run: ollama pull gemma3:4b", type="negative")
            return
        go.props("loading")
        go.disable()
        placeholder.set_visibility(False)
        spinner.set_visibility(True)
        report.set_content("")
        meta.text = ""
        prompt = PROMPT.format(modality=modality.value, clinical=(clinical.value or "").strip() or "Not provided")
        t0, text = time.time(), ""
        try:
            stream = await AsyncClient(host=HOST).chat(
                model=model.value,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt, "images": [state["image"]]},
                ],
                options={"temperature": temp.value},
                stream=True,
            )
            async for chunk in stream:
                spinner.set_visibility(False)
                text += chunk["message"]["content"]
                report.set_content(text)
        except Exception as ex:
            ui.notify(f"Generation failed with '{model.value}': {ex}", type="negative", multi_line=True)
            placeholder.set_visibility(not text)
        else:
            state["text"] = text
            meta.text = f"{model.value} · generated in {time.time() - t0:.1f} s · draft, not reviewed"
        finally:
            spinner.set_visibility(False)
            go.props(remove="loading")
            go.enable()

    def clear():
        state.update(image=None, text="")
        viewer.clear()
        with viewer:
            ui.label("No image loaded").classes("text-grey-6")
        report.set_content("")
        meta.text = ""
        placeholder.set_visibility(True)
        clinical.value = ""

    async def copy():
        if not state["text"]:
            return
        await ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(state['text'])})")
        ui.notify("Report copied", type="positive")

    def download():
        if state["text"]:
            data = state["text"].encode("utf-8")
            if hasattr(ui.download, "content"):  # NiceGUI 3.x
                ui.download.content(data, "report.txt")
            else:  # NiceGUI 2.x
                ui.download(data, "report.txt")

    with ui.column().classes("w-full max-w-6xl mx-auto p-6 gap-4"):
        ui.label("Radiology report drafting").classes("text-2xl font-semibold")
        ui.label("Upload an image, add clinical context, and review the generated draft.").classes("opacity-70 -mt-3")
        ui.label("Research prototype. A qualified radiologist must review every finding before it is used.").classes("notice w-full")

        with ui.row().classes("w-full gap-6 items-start md:flex-nowrap"):
            # ---------- left: input ----------
            with ui.column().classes("w-full md:w-5/12 gap-3"):
                viewer = ui.column().classes("viewer w-full items-center justify-center")
                with viewer:
                    ui.label("No image loaded").classes("text-grey-6")
                upload = ui.upload(on_upload=on_upload, auto_upload=True, max_files=1, label="Drop an image or click to upload")
                upload.props('accept=".png,.jpg,.jpeg" flat bordered color=primary').classes("w-full")

                modality = ui.select(MODALITIES, value=MODALITIES[0], label="Modality").props("outlined dense").classes("w-full")
                clinical = ui.textarea(label="Clinical context (optional)", placeholder="e.g. 65M, cough and fever for 5 days")
                clinical.props("outlined autogrow").classes("w-full")

                default = next((m for m in models if "medgemma" in m), models[0] if models else None)
                model = ui.select(models, value=default, label="Vision model").props("outlined dense").classes("w-full")
                if not models:
                    ui.label("No vision model found on the Ollama server. Run: ollama pull gemma3:4b").classes("text-negative text-sm")
                temp = ui.slider(min=0, max=1, step=0.05, value=0.1).classes("w-full")
                ui.label().bind_text_from(temp, "value", lambda v: f"Temperature {v:.2f}").classes("text-xs opacity-60 -mt-2")

                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Clear", on_click=clear).props("flat")
                    go = ui.button("Generate report", icon="auto_awesome", on_click=generate).props("unelevated")

            # ---------- right: report ----------
            with ui.column().classes("w-full md:w-7/12"):
                with ui.column().classes("paper w-full"):
                    placeholder = ui.label("The draft report will appear here.").classes("placeholder")
                    spinner = ui.spinner("dots", size="lg", color="primary")
                    spinner.set_visibility(False)
                    report = ui.markdown("")
                    meta = ui.label().classes("meta mt-6")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Copy", icon="content_copy", on_click=copy).props("flat dense")
                    ui.button("Download .txt", icon="download", on_click=download).props("flat dense")


ui.run(title="Radiology report drafting", dark=True, reload=False, port=8080)