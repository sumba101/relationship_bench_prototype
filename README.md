# Relationship Bench Prototype

A benchmarking dataset and evaluation framework for comparing how different LLMs respond to real-world romantic relationship advice scenarios sourced from r/relationship_advice on Reddit.

## Goal

The aim is to evaluate and compare LLM outputs on relationship advice posts by having each model provide a reasoned response and classify the situation into one of the following categories:

| Classification | Description |
|---|---|
| **End Relationship** | Advice to break up, divorce, or cut contact |
| **Communicate** | Advice to have an open, honest conversation with the partner |
| **Give Space / Time** | Advice to step back and allow time for reflection |
| **Set / Respect Boundaries** | Advice to establish or enforce personal limits |
| **Seek Therapy / Counselling** | Advice to pursue professional help individually or as a couple |
| **Compromise** | Advice to find a middle ground or mutual agreement |

## Dataset

Located in the `dataset/` folder:

| File | Description |
|---|---|
| `Relationship_Bench_Proto.csv` | Full original dataset (110 posts) with an audit `Comments` column documenting why certain posts were excluded |
| `Relationship_Bench_Filtered.csv` | Filtered working dataset (91 posts) — only established, ongoing romantic or marital relationships |

All posts were sourced from r/relationship_advice between **March and April 2026**. Posts are captured as their original submission — any post where the author had appended an **"Update:"** or **"Edit:"** section was excluded to preserve the unaltered, initial account of the situation, free from any influence of community feedback or subsequent developments.

### Filtering Criteria

Posts were excluded if they:
- Were not about an **established, ongoing romantic or marital relationship** (e.g. sibling disputes, parent-child conflicts, friendships)
- Described a relationship that had **already ended** (breakups, separations, processing an ex)
- Involved situations **too early to be considered established** (under ~2 months, no defined commitment)
- Were **duplicates**

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
