# R3 — RAG vs fine-tune (for a knowledge task)

**Question:** for a knowledge task over the guide, does a fine-tuned 0.5B beat
RAG, and does RAG+tuned beat either? **Skill:** RAG vs weights, product selection.

## Method
Teacher (Llama-3-8B) generates 459 train / 120 test QA from the KB chunks
(short factual answers). Fine-tune the 0.5B closed-book on the train QA. Answer the
120 held-out questions under 4 configs; the 30B judges each vs the gold answer.

## Result (30B-judged, 120 questions)

| config | correct |
|---|---|
| base, closed-book | 7.5% |
| base + RAG | 25.8% |
| tuned, closed-book | 10.0% |
| tuned + RAG | 28.3% |

## Findings
- **RAG beats closed-book fine-tuning 2.6×** (25.8 vs 10.0). Retrieval wins for
  factual recall.
- **Fine-tuning barely helps closed-book** (7.5 → 10.0): a 0.5B learns the answer
  *style* from SFT but not the *facts* — you cannot cheaply memorize a corpus into
  a small model.
- **RAG+tuned is marginally best** (+2.5pt over RAG): retrieval does the work,
  fine-tuning adds format polish.

## Caveat
Absolute accuracies are low because the LLM-generated QA is noisy (ambiguous
questions, gold answers that don't always match). So these *understate* RAG's real
quality (retrieval recall@5 is 93%, R13). The **relative** comparison is the
finding and is unambiguous.

## Pass/fail
Bar: "a stated decision rule." **Pass:** knowledge → RAG, not fine-tuning; combine
for a small edge.

## The Part VII decision framework (now complete)
| task type | build | evidence |
|---|---|---|
| convention / format | fine-tune (custom SLM) | ch.33/34 |
| open-ended judgement | rent the big model | R2 |
| knowledge / facts | RAG | R3 |

Artifacts: `rag/r3_qa_gen.py`, `rag/r3_train.py`, `rag/r3_answer.py`, `rag/r3_judge.py`.
