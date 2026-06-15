from src.cleaner import clean_markdown


def test_cleaner_removes_extra_blank_lines() -> None:
    text = "第一段\n\n\n\n第二段"
    assert clean_markdown(text) == "第一段\n\n第二段"


def test_cleaner_keeps_headings_and_list() -> None:
    text = "# 标题  \n\n- 项目1  \n- 项目2"
    cleaned = clean_markdown(text)
    assert "# 标题" in cleaned
    assert "- 项目1" in cleaned
    assert "- 项目2" in cleaned


def test_cleaner_does_not_empty_text() -> None:
    text = "<div>内容</div>\n\n正文"
    cleaned = clean_markdown(text)
    assert "正文" in cleaned


def test_cleaner_removes_url_noise_but_keeps_link_text() -> None:
    text = (
        "# 标题\n\n"
        "参考 [牧原公告](https://example.com/report?id=1) 和 https://example.com/raw。\n\n"
        "![图表说明](https://example.com/chart.png)\n\n"
        "[ref]: https://example.com/ref"
    )
    cleaned = clean_markdown(text)
    assert "牧原公告" in cleaned
    assert "图表说明" in cleaned
    assert "https://example.com" not in cleaned
    assert "[ref]:" not in cleaned
