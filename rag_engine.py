"""
RAG layer: TF-IDF retrieval over the compatibility rows + an LLM answer written from those rows.

The structured rows are the ground truth; the LLM only explains them. The system
prompt forbids stating compatibility that is not in the retrieved rows.
TF-IDF (scikit-learn) is plenty for a few hundred short rows; swap in a vector DB
if the data grows into the tens of thousands.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are a mainframe (z/OS) product-compatibility assistant for z/OS migrations.
You receive ROWS from a compatibility matrix. They are ground truth:
- 'R3.2', 'z/OS 3.1' and 'z/OS 2.5' are each Yes / No / In Progress / Unverified. Unverified means NO vendor statement was found - never present it as compatible.
- 'PTF' says whether the vendor lists PTFs needed for z/OS 3.2.
Rules:
- Answer ONLY from the rows. If they do not answer the question, say so and point to the vendor link in the rows.
- Never invent products, versions, PTF numbers or dates.
- Group by vendor. Call out products that are Yes on an older z/OS (2.5 or 3.1) but No on a newer one, and any that need PTFs.
- End with one line telling the user to confirm against the live vendor page before production changes, and
  include the source links from the rows you used.
- Be concise: a short intro, then a clean grouped list."""


@dataclass
class RagIndex:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    matrix: object

    @classmethod
    def build(cls, df: pd.DataFrame) -> "RagIndex":
        d = df.reset_index(drop=True)
        texts = d.apply(
            lambda r: (f"{r['vendor']} {r['product']} version {r['vrm']} z/OS 3.2 {r['zos32']} z/OS 3.1 {r['zos31']} z/OS 2.5 {r['zos25']} "
                       f"PTF {r['ptf_required']} EOS {r['eos_date']} latest {r['latest_version']} {r['comments']}"),
            axis=1,
        ).tolist()
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        return cls(df=d, vectorizer=vec, matrix=vec.fit_transform(texts))

    def retrieve(self, query: str, top_k: int = 12) -> pd.DataFrame:
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        idx = [i for i in sims.argsort()[::-1][:top_k] if sims[i] > 0]
        return self.df.iloc[idx]


def _rows_text(rows: pd.DataFrame) -> str:
    return "\n".join(
        f"- {r['vendor']} | {r['product']} {r['vrm']} | z/OS 2.5: {r['zos25']} | z/OS 3.1: {r['zos31']} | R3.2: {r['zos32']} | "
        f"PTF: {r['ptf_required']} | EOS: {r['eos_date']} | latest: {r['latest_version']} | "
        f"{r['comments']} | source: {r['link']} ({r['data_status']})"
        for _, r in rows.iterrows()
    )


def generate_answer(question: str, rows: pd.DataFrame, api_key: str | None = None, model: str | None = None) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    body = _rows_text(rows) if not rows.empty else "(no matching rows)"
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\nMATRIX ROWS:\n{body}"}],
    )
    return "".join(b.text for b in resp.content if hasattr(b, "text"))
