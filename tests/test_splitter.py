from ragnarok.splitter import sections, split


def words(n: int, word: str = "w") -> str:
    return " ".join([word] * n)


def test_headings_inside_code_fences_do_not_split():
    body = "# A\ntext\n```bash\n# not a heading\n```\n## B\nmore"
    assert [s.path for s in sections(body)] == [("A",), ("A", "B")]


def test_h4_does_not_split():
    body = "# A\n#### deep\ntext"
    assert [s.path for s in sections(body)] == [("A",)]


def test_small_sections_join_with_the_next_under_the_same_parent():
    body = f"# T\n## A\n{words(10)}\n## B\n{words(10)}\n## C\n{words(100)}"
    chunks = split("Note", body, min_words=80, max_words=300)
    assert len(chunks) == 1
    assert chunks[0].heading == "Note > T"
    assert "A" in chunks[0].text and "B" in chunks[0].text


def test_top_level_sections_are_siblings():
    body = f"# One\n{words(100)}\n# Two\n{words(10)}\n# Three\n{words(100)}"
    chunks = split("N", body, min_words=80)
    assert [c.heading for c in chunks] == ["N > One", "N > Two"]
    assert chunks[1].text.startswith("w") and "Three" in chunks[1].text


def test_small_section_does_not_join_into_another_branch():
    body = f"# T\n## A\n{words(100)}\n### x\n{words(10)}\n## B\n{words(100)}"
    headings = [c.heading for c in split("N", body, min_words=80)]
    assert headings == ["N > T", "N > T > A > x", "N > T > B"]  # empty T joins A


def test_small_last_section_joins_the_one_before():
    body = f"# T\n## A\n{words(100)}\n## B\n{words(5)}"
    chunks = split("N", body, min_words=80)
    assert len(chunks) == 1 and chunks[0].heading == "N > T"
    assert chunks[0].text.endswith("B\nw w w w w")


def test_large_sections_split_at_paragraphs():
    body = "# A\n" + "\n\n".join(words(120) for _ in range(5))
    chunks = split("N", body, max_words=300)
    assert [len(c.text.split()) for c in chunks] == [240, 240, 120]
    assert all(c.heading == "N > A" for c in chunks)


def test_long_paragraph_splits_by_words():
    chunks = split("N", "# A\n" + words(700), max_words=300)
    assert [len(c.text.split()) for c in chunks] == [300, 300, 100]


def test_min_zero_keeps_each_heading():
    body = "# A\nx\n# B\ny"
    assert [c.heading for c in split("N", body, min_words=0)] == ["N > A", "N > B"]


def test_description_chunk():
    chunks = split("N", "# A\ntext", description="What it is.", project="p", area="a")
    assert chunks[-1].heading == "N > description"
    assert chunks[-1].text == "project: p. area: a. What it is."
    assert [c.ord for c in chunks] == [0, 1]


def test_note_without_headings():
    chunks = split("N", words(50))
    assert len(chunks) == 1 and chunks[0].heading == "N"
