"""Local FastAPI backend; not an authenticated public deployment."""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.pipeline import ResearchAssistant


class Question(BaseModel):
    """Question and optional multi-paper filter."""

    question: str = Field(min_length=1, max_length=2000)
    paper_id: str | None = None


def create_app(config_path: str = "configs/fast_cpu.yaml") -> FastAPI:
    """Construct an independently configurable application."""
    assistant = ResearchAssistant(config_path)
    app = FastAPI(title="Research Paper Assistant", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "device": assistant.device}

    @app.get("/papers")
    def papers() -> list[dict]:
        return assistant.papers()

    @app.post("/papers")
    def upload(file: UploadFile = File(...)) -> dict:
        limit = assistant.config["server"]["max_upload_mb"] * 1024 * 1024
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / Path(file.filename or "paper.pdf").name
            with path.open("wb") as target:
                size = 0
                while block := file.file.read(1024 * 1024):
                    size += len(block)
                    if size > limit:
                        raise HTTPException(413, "PDF exceeds upload limit")
                    target.write(block)
            try:
                result = assistant.ingest(path)
                return {k: v for k, v in result.items() if k not in {"chunks", "passages"}}
            except ValueError as error:
                raise HTTPException(422, str(error)) from error

    @app.post("/ask")
    def ask(question: Question) -> dict:
        try:
            return assistant.ask(question.question, question.paper_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/papers/{paper_id}/summary")
    def summary(paper_id: str) -> dict:
        try:
            return assistant.summarize(paper_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/papers/{paper_id}/findings")
    def findings(paper_id: str) -> list[dict]:
        try:
            return assistant.findings(paper_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/papers/{paper_id}/notes")
    def notes(paper_id: str) -> dict:
        try:
            return {"markdown": assistant.notes(paper_id)}
        except ValueError as error:
            raise HTTPException(404, str(error)) from error

    return app


def main() -> None:
    """Run on configured loopback address by default."""
    import uvicorn

    config = os.environ.get("RPA_CONFIG", "configs/fast_cpu.yaml")
    app = create_app(config)
    server = ResearchAssistant(config).config["server"]
    uvicorn.run(app, host=server["host"], port=server["api_port"])


if __name__ == "__main__":
    main()
