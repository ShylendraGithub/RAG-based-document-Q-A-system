"""An extractive stand-in for the LLM so the project runs offline.

This is NOT an AI model and does not generate text. It reads the same numbered <document>
passages a real model would receive, picks the one or two sentences that share the most terms
with the question, and returns them with their [n] citation. If nothing overlaps enough it returns
the standard "I don't know" sentence. It exists to demo the pipeline and to give tests a
deterministic model; it cannot paraphrase, combine facts or do any reasoning.
"""

from __future__ import annotations

import math
import re

from .chunker import split_sentences
from .text import content_terms

NO_ANSWER = "I don't know based on the provided documents."
_DOC = re.compile(r'<document index="(\d+)"[^>]*>\n(.*?)\n</document>', re.DOTALL)


class DemoLLM:
    def complete(self, system: str, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        question = content.rsplit("Question:", 1)[-1].strip()
        q_terms = set(content_terms(question))
        if not q_terms:
            return NO_ANSWER
        needed = min(len(q_terms), max(2, math.ceil(0.4 * len(q_terms))))

        scored: list[tuple[int, int, int, str]] = []  # (overlap, -doc_index, -sentence_pos, text/idx)
        for m in _DOC.finditer(content):
            index = int(m.group(1))
            for pos, raw in enumerate(split_sentences(m.group(2))):
                sentence = re.sub(r"^\s*(?:[-*•+]|\d{1,3}[.)])\s+", "", raw).strip()
                overlap = len(q_terms & set(content_terms(sentence)))
                if overlap >= needed:
                    scored.append((overlap, -index, -pos, f"{index}\x00{sentence}"))
        if not scored:
            return NO_ANSWER
        scored.sort(reverse=True)
        best = scored[0][0]
        picked = [s for s in scored if s[0] >= max(needed, math.ceil(best * 0.8))][:2]
        parts = []
        for _, _, _, packed in picked:
            index, sentence = packed.split("\x00", 1)
            parts.append(f"{sentence.rstrip()} [{index}]")
        return " ".join(parts)
