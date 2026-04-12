"""
Relationship Bench — LLM Classification Runner

Runs all configured models in parallel (batch size 25 concurrent requests per model,
all models concurrently) and writes results to results/<model_slug>/<timestamp>.csv.

Retries up to MAX_RETRIES times on JSON schema validation failures before giving up.

Usage:
    # Full run, all models
    python classify.py

    # Subset of models
    python classify.py --models claude-haiku-4-5-20251001,gpt-5.4-mini

    # Smoke test — first N posts only
    python classify.py --limit 3 --models gpt-5.4-mini

    # Fix error rows in an existing results CSV in place
    python classify.py --retry-errors results/claude-haiku-4-5-20251001/20260413_001551.csv --models claude-haiku-4-5-20251001
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

BATCH_SIZE = 25   # max concurrent requests per model
MAX_RETRIES = 3   # retries on JSON schema validation failure only
DATASET_PATH = Path(__file__).parent / "dataset" / "Relationship_Bench_Filtered.csv"
RESULTS_DIR  = Path(__file__).parent / "results"

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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")


def _extract_json(content: str) -> str:
    """Strip markdown code fences if the model wraps its JSON output."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
    return match.group(1) if match else content


# ─────────────────────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────────────────────

async def classify_post(
    semaphore: asyncio.Semaphore,
    model: str,
    row: dict,
) -> dict:
    async with semaphore:
        last_exc: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
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
                if attempt > 1:
                    print(f"[{model}] ID {row['ID']} succeeded on attempt {attempt}")
                return {
                    "ID": row["ID"],
                    "Link": row["Link"],
                    "Post Content": row["Post Content"],
                    "Classification": result.classification,
                    "Reasoning": result.reasoning,
                    "Error": "",
                }
            except litellm.JSONSchemaValidationError as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    print(f"[{model}] ID {row['ID']} JSON validation failed — retrying ({attempt}/{MAX_RETRIES}) ...")
                    continue
            except Exception as exc:
                # Non-retryable (auth, network, etc.) — fail immediately
                return {
                    "ID": row["ID"],
                    "Link": row["Link"],
                    "Post Content": row["Post Content"],
                    "Classification": "",
                    "Reasoning": "",
                    "Error": str(exc),
                }

        return {
            "ID": row["ID"],
            "Link": row["Link"],
            "Post Content": row["Post Content"],
            "Classification": "",
            "Reasoning": "",
            "Error": f"Failed after {MAX_RETRIES} attempts: {last_exc}",
        }


async def run_model(model: str, rows: list[dict], timestamp: str) -> None:
    slug = _model_slug(model)
    model_dir = RESULTS_DIR / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{timestamp}.csv"

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    tasks = [classify_post(semaphore, model, row) for row in rows]
    print(f"[{model}] Starting {len(tasks)} posts ...")
    results = await asyncio.gather(*tasks)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ID", "Link", "Post Content", "Classification", "Reasoning", "Error"]
        )
        writer.writeheader()
        writer.writerows(results)

    errors = sum(1 for r in results if r["Error"])
    print(f"[{model}] Done — {len(results) - errors}/{len(results)} OK  ->  {out_path}")


async def retry_errors(csv_path: Path, model: str) -> None:
    df = pd.read_csv(csv_path)
    error_mask = df["Error"].notna() & (df["Error"].astype(str).str.strip() != "")
    n_errors = error_mask.sum()

    if n_errors == 0:
        print("No error rows found — nothing to retry.")
        return

    print(f"Found {n_errors} error row(s) in {csv_path}. Retrying with [{model}] ...")

    # Pull Post Content from the dataset for the failed IDs
    dataset = pd.read_csv(DATASET_PATH)[["ID", "Link", "Post Content"]]
    error_ids = df.loc[error_mask, "ID"].tolist()
    rows = dataset[dataset["ID"].isin(error_ids)].to_dict(orient="records")

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    tasks = [classify_post(semaphore, model, row) for row in rows]
    results = await asyncio.gather(*tasks)

    # Merge results back into the original DataFrame
    for res in results:
        mask = df["ID"] == res["ID"]
        df.loc[mask, "Classification"] = res["Classification"]
        df.loc[mask, "Reasoning"]      = res["Reasoning"]
        df.loc[mask, "Error"]          = res["Error"] if res["Error"] else float("nan")

    df.to_csv(csv_path, index=False)
    still_errors = sum(1 for r in results if r["Error"])
    print(f"Done — {n_errors - still_errors}/{n_errors} fixed. CSV updated in place: {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

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
        help="Comma-separated exact model IDs to run (default: all configured models)",
    )
    parser.add_argument(
        "--retry-errors",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Path to an existing results CSV; reruns error rows and updates the file in place. Requires --models with exactly one model.",
    )
    args = parser.parse_args()

    # ── Retry-errors mode ────────────────────────────────────────────────────
    if args.retry_errors is not None:
        csv_path = Path(args.retry_errors)
        if not csv_path.exists():
            parser.error(f"CSV not found: {csv_path}")
        if not args.models or len(args.models.split(",")) != 1:
            parser.error("--retry-errors requires exactly one model via --models")
        model = args.models.strip()
        if model not in MODELS:
            parser.error(f"Unknown model: {model!r}\nAvailable:\n  " + "\n  ".join(MODELS))
        await retry_errors(csv_path, model)
        return

    # ── Normal run ───────────────────────────────────────────────────────────
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    await asyncio.gather(*[run_model(model, rows, timestamp) for model in models])
    print(f"\nAll models complete.")


if __name__ == "__main__":
    asyncio.run(main())
