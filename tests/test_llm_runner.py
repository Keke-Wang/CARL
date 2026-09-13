from carl_mt.data import TranslationExample
from carl_mt.llm_runner import extract_completion, prompt_sha256, render_prompt


def test_prompt_and_response_parsing_are_explicit() -> None:
    template = "Title: {title}\nContext: {context}\nTranslation:"
    example = TranslationExample("标题", "语境", "reference", 7)
    prompt = render_prompt(template, example)
    assert "标题" in prompt and "语境" in prompt
    assert len(prompt_sha256(template)) == 64
    response = {"choices": [{"message": {"content": "  A translation.  "}}]}
    assert extract_completion(response) == "A translation."
