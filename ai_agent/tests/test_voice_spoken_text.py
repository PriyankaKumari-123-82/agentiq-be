"""Unit tests for the voice pipeline's spoken-text derivation.

The TTS voice should speak conversational prose only: markdown syntax is
stripped, pipe tables / charts / artifacts / code become short spoken notices,
and the total length is capped so a single answer can't produce minutes of
audio.
"""
import pytest

from ai_agent.api.v1.voice import (
    _derive_spoken_text,
    _strip_markdown_for_speech,
    SPOKEN_CHAR_CAP,
    TABLE_NOTE,
    CHART_NOTE,
    FILE_NOTE,
    CODE_NOTE,
    TRUNCATION_NOTE,
)
from ai_agent.core.response_schemas import (
    BlockType,
    ContentBlock,
    TableBlock,
    TableColumn,
    ChartBlock,
)


def md_block(text: str) -> ContentBlock:
    return ContentBlock(type=BlockType.MARKDOWN, content=text)


class TestStripMarkdownForSpeech:
    def test_plain_prose_passes_through(self):
        text, flags = _strip_markdown_for_speech("Hello there. All good.")
        assert text == "Hello there. All good."
        assert flags == set()

    def test_headings_and_emphasis_stripped(self):
        text, _ = _strip_markdown_for_speech(
            "## Revenue summary\n\nRevenue **grew** by *12%* in `Q4`."
        )
        assert "##" not in text and "**" not in text and "`" not in text
        assert "Revenue summary" in text
        assert "grew" in text and "12%" in text and "Q4" in text

    def test_links_become_link_text(self):
        text, _ = _strip_markdown_for_speech("See [the dashboard](https://x.io/d).")
        assert text == "See the dashboard."

    def test_pipe_table_lines_removed_and_flagged(self):
        md = (
            "Here are the results:\n"
            "| name | total |\n"
            "| --- | --- |\n"
            "| Acme | 42 |\n"
            "That's the summary."
        )
        text, flags = _strip_markdown_for_speech(md)
        assert "|" not in text
        assert "Acme" not in text
        assert "Here are the results:" in text
        assert "That's the summary." in text
        assert "table" in flags

    def test_fenced_code_removed_and_flagged(self):
        md = "Run this:\n```sql\nSELECT 1;\n```\nDone."
        text, flags = _strip_markdown_for_speech(md)
        assert "SELECT" not in text
        assert "```" not in text
        assert "Run this:" in text and "Done." in text
        assert "code" in flags


class TestDeriveSpokenText:
    def test_markdown_only(self):
        spoken = _derive_spoken_text([md_block("The answer is 42.")])
        assert spoken == "The answer is 42."

    def test_table_block_becomes_notice_once(self):
        table = TableBlock(
            title="t",
            columns=[TableColumn(name="a"), TableColumn(name="b")],
            rows=[["1", "2"]],
        )
        blocks = [
            md_block("Here you go."),
            ContentBlock(type=BlockType.TABLE, content=table),
            ContentBlock(type=BlockType.TABLE, content=table),
        ]
        spoken = _derive_spoken_text(blocks)
        assert spoken.count(TABLE_NOTE) == 1
        assert "Here you go." in spoken

    def test_chart_and_artifact_notices(self):
        chart = ChartBlock(chart_type="bar", title="c", labels=["a"], datasets=[])
        blocks = [
            md_block("Summary."),
            ContentBlock(type=BlockType.CHART, content=chart),
            ContentBlock(
                type=BlockType.ARTIFACT,
                content={"filename": "report.pdf", "download_url": "/x"},
            ),
        ]
        spoken = _derive_spoken_text(blocks)
        assert CHART_NOTE in spoken
        assert FILE_NOTE in spoken

    def test_inline_markdown_table_becomes_notice(self):
        md = "Data below:\n| a | b |\n| - | - |\n| 1 | 2 |"
        spoken = _derive_spoken_text([md_block(md)])
        assert TABLE_NOTE in spoken
        assert "|" not in spoken

    def test_code_note_from_fenced_block(self):
        spoken = _derive_spoken_text([md_block("Try:\n```py\nprint(1)\n```")])
        assert CODE_NOTE in spoken

    def test_length_cap(self):
        long_text = "word " * 1000
        spoken = _derive_spoken_text([md_block(long_text)])
        assert len(spoken) <= SPOKEN_CHAR_CAP + len(TRUNCATION_NOTE) + 5
        assert spoken.endswith(TRUNCATION_NOTE)

    def test_empty_blocks_fall_back_to_generic_line(self):
        spoken = _derive_spoken_text([])
        assert spoken  # never empty — TTS always has something to say
