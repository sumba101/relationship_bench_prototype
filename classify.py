"""
Relationship Bench — LLM Classification Runner

Runs all configured models in parallel (batch size 25 concurrent requests per model,
all models concurrently) and writes one CSV of results per model into a timestamped
results/ subdirectory.
"""

import argparse
import asyncio
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import litellm
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

# Load .env from same directory as this script
load_dotenv(Path(__file__).parent / ".env")

# Enforce that the model's JSON response matches the Pydantic schema
litellm.enable_json_schema_validation = True

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODELS: list[str] = [
    "gemini/gemini-3.1-pro-preview",
    "gemini/gemini-3.1-flash-lite-preview",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "gpt-5.4",
    "gpt-5.4-mini",
]

BATCH_SIZE = 25  # max concurrent requests per model
DATASET_PATH = Path(__file__).parent / "dataset" / "Relationship_Bench_Filtered.csv"
RESULTS_DIR = Path(__file__).parent / "results"

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class ClassificationOutput(BaseModel):
    reasoning: str
    classification: Literal[
        "End Relationship",
        "Communicate",
        "Give Space / Time",
        "Set / Respect Boundaries",
        "Seek Therapy / Counselling",
        "Compromise",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert relationship counsellor evaluating posts from r/relationship_advice.

Read the post and classify the situation into exactly one of the following six categories \
based on what advice is most appropriate for the poster:

- End Relationship: Advice to break up, divorce, or cut contact
- Communicate: Advice to have an open, honest conversation with the partner
- Give Space / Time: Advice to step back and allow time for reflection
- Set / Respect Boundaries: Advice to establish or enforce personal limits
- Seek Therapy / Counselling: Advice to pursue professional help individually or as a couple
- Compromise: Advice to find a middle ground or mutual agreement

Think through the situation carefully, then provide your classification.\
"""

# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(content: str) -> str:
    """Strip markdown code fences if the model wraps its JSON output."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    return match.group(1) if match else content


async def classify_post(
    semaphore: asyncio.Semaphore,
    model: str,
    row: dict,
) -> dict:
    async with semaphore:
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(row["Post Content"])},
                ],
                response_format=ClassificationOutput,
            )
            content = response.choices[0].message.content
            result = ClassificationOutput.model_validate_json(_extract_json(content))
            return {
                "ID": row["ID"],
                "Link": row["Link"],
                "Classification": result.classification,
                "Reasoning": result.reasoning,
                "Error": "",
            }
        except Exception as exc:
            return {
                "ID": row["ID"],
                "Link": row["Link"],
                "Classification": "",
                "Reasoning": "",
                "Error": str(exc),
            }


async def run_model(model: str, rows: list[dict], run_dir: Path) -> None:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
    out_path = run_dir / f"{slug}.csv"
    semaphore = asyncio.Semaphore(BATCH_SIZE)

    tasks = [classify_post(semaphore, model, row) for row in rows]
    print(f"[{model}] Starting {len(tasks)} posts ...")
    results = await asyncio.gather(*tasks)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "Link", "Classification", "Reasoning", "Error"]
        )
        writer.writeheader()
        writer.writerows(results)

    errors = sum(1 for r in results if r["Error"])
    print(f"[{model}] Done — {len(results) - errors}/{len(results)} OK  ->  {out_path.name}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Relationship Bench — LLM Classification Runner")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only classify the first N posts (useful for smoke-testing before a full run)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        metavar="MODEL[,MODEL...]",
        help="Comma-separated list of models to run (default: all). Substring match accepted, e.g. 'mini,haiku'",
    )
    args = parser.parse_args()

    models = MODELS
    if args.models is not None:
        requested = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in requested if m not in MODELS]
        if unknown:
            parser.error(f"Unknown model(s): {unknown}\nAvailable:\n  " + "\n  ".join(MODELS))
        models = [m for m in MODELS if m in requested]
        print(f"--models: {models}\n")

    df = pd.read_csv(DATASET_PATH)
    total_posts = len(df)
    if args.limit is not None:
        df = df.head(args.limit)
        print(f"--limit {args.limit}: running on {len(df)} of {total_posts} posts\n")

    rows = df[["ID", "Link", "Post Content"]].to_dict(orient="records")

    RESULTS_DIR.mkdir(exist_ok=True)
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir()
    print(f"Run directory: {run_dir}\n")

    await asyncio.gather(*[run_model(model, rows, run_dir) for model in models])
    print(f"\nAll models complete. Results saved to: {run_dir}")


if __name__ == "__main__":
    asyncio.run(main())
