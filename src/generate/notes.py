"""Generate portable study notes without implying model-generated claims are facts."""


def render_notes(paper: str, summary: dict, findings: list[dict]) -> str:
    """Render Markdown with source quotes and citation metadata."""
    lines = [
        f"# Study notes: {paper}",
        "",
        "> Verify every generated claim against the PDF.",
        "",
        "## Generated summary",
        "",
        summary["summary"],
        "",
        summary["warning"],
        "",
        "## Summary source audit",
    ]
    for mapped in summary["map_summaries"]:
        source = mapped["source"]
        lines.extend(
            [
                "",
                mapped["summary"],
                f"Source: {source['paper']} — {source['section']}, p. {source['page']}",
                "",
                "> " + source["passage"],
            ]
        )
    for finding in findings:
        lines.extend(["", f"## {finding['field'].title()}", "", finding["status"]])
        for candidate in finding["evidence"]:
            source = candidate["source"]
            lines.extend(
                [
                    "",
                    "> " + candidate["text"],
                    f"{source['paper']} — {source['section']}, p. {source['page']}",
                ]
            )
    lines.extend(
        [
            "",
            "## Review questions",
            "",
            "- What evidence supports the main claim?",
            "- What are the evaluation assumptions and failure cases?",
            "- Which results need independent replication?",
            "",
        ]
    )
    return "\n".join(lines)
