"""Create a disclosed assistant-authored study set from two open-access sample PDFs.

Questions were written after inspecting source passages, before running predictions.
This is a convenience sample, not independent human labels or a held-out benchmark.
"""

import csv
import json
from pathlib import Path

from src.pipeline import ResearchAssistant

# Each entry links a question and reference span to a specific inspected PDF passage.
ATTENTION = [
    (11, "What network architecture does Attention Is All You Need propose?", "Transformer"),
    (11, "What BLEU score does the abstract report for English-to-German translation?", "28.4"),
    (17, "Which GPU model is named in the Introduction's twelve-hour training claim?", "P100"),
    (19, "What other name does the Background give for self-attention?", "intra-attention"),
    (25, "How many identical layers are in the Transformer encoder stack?", "6"),
    (25, "What normalization follows the encoder residual connections?", "layer normalization"),
    (26, "How many identical layers are in the Transformer decoder stack?", "6"),
    (31, "Which function converts scaled query-key dot products into weights?", "softmax"),
    (44, "How many parallel attention heads are used in the described model?", "8"),
    (49, "Which activation separates the two feed-forward linear transformations?", "ReLU"),
    (51, "What is the inner-layer dimensionality of the feed-forward network?", "2048"),
    (56, "Which functions define the fixed positional encodings?", "sine and cosine"),
    (
        69,
        "About how many sentence pairs are in the English-German training dataset?",
        "4.5 million",
    ),
    (69, "How many sentences are in the English-French training dataset?", "36M"),
    (70, "How many NVIDIA P100 GPUs were used to train the models?", "8"),
    (70, "For how many steps were the base models trained?", "100,000"),
    (71, "Which optimizer was used to train the Transformer?", "Adam"),
    (73, "How many learning-rate warmup steps were used?", "4000"),
    (90, "What beam size was used for machine translation inference?", "4"),
]
BERT = [
    (2, "What does BERT stand for?", "Bidirectional Encoder Representations from Transformers"),
    (
        2,
        "How many additional output layers are needed to fine-tune BERT "
        "as described in the abstract?",
        "one",
    ),
    (
        4,
        "How many NLP tasks does the BERT abstract report new state-of-the-art results on?",
        "eleven",
    ),
    (4, "What GLUE score is reported in the BERT abstract?", "80.5%"),
    (4, "What MultiNLI accuracy is reported in the BERT abstract?", "86.7%"),
    (
        7,
        "What pre-training task is used in addition to the masked language model?",
        "next sentence prediction",
    ),
    (
        37,
        "What type of Transformer is BERT's model architecture?",
        "multi-layer bidirectional Transformer encoder",
    ),
    (39, "How many total parameters does BERTBASE have?", "110M"),
    (39, "How many total parameters does BERTLARGE have?", "340M"),
]


def main() -> None:
    """Validate every reference span and save concise source excerpts with true pages."""
    assistant = ResearchAssistant("configs/fast_cpu.yaml")
    rows = []
    for filename, specifications in [
        ("1706.03762v7.pdf", ATTENTION),
        ("1810.04805v2.pdf", BERT),
    ]:
        paper = assistant.ingest(Path(assistant.config["paths"]["papers"]) / filename)
        for index, question, answer in specifications:
            passage = paper["passages"][index]
            offset = passage["text"].index(answer)
            # A short literal excerpt preserves provenance without publishing full papers.
            start = max(0, offset - 90)
            stop = min(len(passage["text"]), offset + len(answer) + 90)
            rows.append(
                {
                    "id": f"assistant-{len(rows) + 1:02}",
                    "paper_id": paper["paper_id"],
                    "question": question,
                    "reference_answers_json": json.dumps([answer]),
                    "evidence_passage": passage["text"][start:stop],
                    "section": passage["section"],
                    "pdf_page": passage["page"],
                    "unanswerable": "false",
                    "label_author": "assistant",
                }
            )
        rows.append(
            {
                "id": f"assistant-{len(rows) + 1:02}",
                "paper_id": paper["paper_id"],
                "question": "What is the capital of Madagascar?",
                "reference_answers_json": json.dumps([""]),
                "evidence_passage": "",
                "section": "",
                "pdf_page": "",
                "unanswerable": "true",
                "label_author": "assistant",
            }
        )
    assert len(rows) == 30
    path = Path("data/eval/assistant_questions.csv")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} assistant-authored labels to {path}")


if __name__ == "__main__":
    main()
