"""Gradio UI backed by the same cached pipeline as the HTTP service."""

import html
import os
import tempfile
from pathlib import Path

import gradio as gr

from src.pipeline import ResearchAssistant


def highlighted_sources(result: dict) -> str:
    """Escape PDF text before highlighting literal answer spans to avoid HTML injection."""
    fragments = []
    for source in result["sources"]:
        passage = html.escape(source["passage"])
        answer = html.escape(result["answer"])
        if not result["abstained"] and answer:
            passage = passage.replace(answer, f"<mark>{answer}</mark>")
        title = html.escape(f"{source['paper']} — {source['section']}, p. {source['page']}")
        fragments.append(f"<details open><summary>{title}</summary><p>{passage}</p></details>")
    return "\n".join(fragments)


def build_ui(config_path: str) -> gr.Blocks:
    """Create upload, evidence, summary, findings and download views."""
    assistant = ResearchAssistant(config_path)

    def selected(paper_id: str | None) -> str:
        if not paper_id:
            raise gr.Error("Select an uploaded paper first.")
        return paper_id

    with gr.Blocks(title="Research Paper Assistant") as demo:
        gr.Markdown(
            "# Research Paper Assistant\nCPU-friendly research tools. "
            "Generated text may be wrong. Verify every claim against its source."
        )
        upload = gr.File(label="Upload a born-digital PDF", file_types=[".pdf"], type="filepath")
        selector = gr.Dropdown(
            choices=[(p["paper"], p["paper_id"]) for p in assistant.papers()],
            label="Paper (leave blank to search all)",
        )
        status = gr.Textbox(label="Upload status")

        def ingest(path: str) -> tuple:
            if not path:
                raise gr.Error("Choose a PDF")
            try:
                record = assistant.ingest(path)
            except ValueError as error:
                raise gr.Error(str(error)) from error
            return gr.Dropdown(
                choices=[(p["paper"], p["paper_id"]) for p in assistant.papers()],
                value=record["paper_id"],
            ), "Ready; cached=" + str(record["cached"])

        upload.upload(ingest, upload, [selector, status])
        with gr.Tab("Q&A"):
            question = gr.Textbox(label="Question")
            answer = gr.JSON(label="Answer and heuristic support score")
            evidence = gr.HTML(label="Source passages")

            def ask(text: str, paper_id: str | None) -> tuple:
                if not text.strip():
                    raise gr.Error("Enter a question first.")
                try:
                    result = assistant.ask(text, paper_id)
                except ValueError as error:
                    raise gr.Error(str(error)) from error
                return result, highlighted_sources(result)

            gr.Button("Ask").click(ask, [question, selector], [answer, evidence])
        with gr.Tab("Summary"):
            summary = gr.JSON(label="Generated summary and source audit")

            def summarize_paper(paper_id: str | None) -> dict:
                try:
                    return assistant.summarize(selected(paper_id))
                except (ValueError, RuntimeError) as error:
                    raise gr.Error(str(error)) from error

            gr.Button("Summarize").click(summarize_paper, selector, summary)
        with gr.Tab("Findings"):
            findings = gr.Dataframe(
                headers=["Field", "Status", "Candidate source sentences"],
                datatype=["str", "str", "str"],
                interactive=False,
            )

            def extract(paper_id: str) -> list[list[str]]:
                paper_id = selected(paper_id)
                return [
                    [
                        f["field"],
                        f["status"],
                        "\n".join(
                            f"{e['text']} ({e['source']['paper']}, {e['source']['section']}, "
                            f"p. {e['source']['page']})"
                            for e in f["evidence"]
                        ),
                    ]
                    for f in assistant.findings(paper_id)
                ]

            gr.Button("Extract candidates").click(extract, selector, findings)
        with gr.Tab("Study notes"):
            download = gr.File(label="Download Markdown notes")

            def notes(paper_id: str) -> str:
                paper_id = selected(paper_id)
                directory = Path(tempfile.mkdtemp(prefix="rpa-notes-"))
                path = directory / "study-notes.md"
                path.write_text(assistant.notes(paper_id))
                return str(path)

            gr.Button("Generate notes").click(notes, selector, download)
    return demo


def main() -> None:
    """Launch locally, with a single model operation in flight."""
    config_path = os.environ.get("RPA_CONFIG", "configs/fast_cpu.yaml")
    assistant = ResearchAssistant(config_path)
    server = assistant.config["server"]
    build_ui(config_path).queue(default_concurrency_limit=1).launch(
        server_name=server["host"],
        server_port=server["ui_port"],
        share=False,
        max_file_size=f"{server['max_upload_mb']}mb",
    )


if __name__ == "__main__":
    main()
