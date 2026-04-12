"""
Relationship Bench — Cost Estimator

Uses tiktoken to count tokens in the actual prompt + dataset content,
then prints a per-model cost breakdown before you commit to a full run.

Usage:
    python estimate_costs.py
"""

from pathlib import Path

import pandas as pd
import tiktoken

# ─────────────────────────────────────────────────────────────────────────────
# Pricing  (input $/MTok, output $/MTok)
# ─────────────────────────────────────────────────────────────────────────────

PRICING: dict[str, tuple[float, float]] = {
    "gemini/gemini-3.1-pro-preview":        (2.00,  12.00),
    "gemini/gemini-3.1-flash-lite-preview": (0.25,   1.50),
    "claude-sonnet-4-6":                    (3.00,  15.00),
    "claude-haiku-4-5-20251001":            (1.00,   5.00),
    "gpt-5.4":                              (2.50,  15.00),
    "gpt-5.4-mini":                         (0.75,   4.50),
}

# Must stay in sync with classify.py
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

# Conservative estimate: reasoning ~150 tokens + JSON structure/classification ~50 tokens
EST_OUTPUT_TOKENS = 200

DATASET_PATH = Path(__file__).parent / "dataset" / "Relationship_Bench_Filtered.csv"


def main() -> None:
    # cl100k_base is a good cross-provider approximation (used by GPT-4 family)
    enc = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv(DATASET_PATH)
    posts = df["Post Content"].dropna().tolist()
    n = len(posts)

    system_tokens = len(enc.encode(SYSTEM_PROMPT))
    post_token_counts = [len(enc.encode(str(p))) for p in posts]
    avg_post_tokens = sum(post_token_counts) / len(post_token_counts)
    total_post_tokens = sum(post_token_counts)

    total_input_per_model = total_post_tokens + system_tokens * n
    avg_input_per_call = system_tokens + avg_post_tokens

    print("=" * 70)
    print("Relationship Bench — Cost Estimate")
    print("=" * 70)
    print(f"  Posts              : {n}")
    print(f"  System prompt      : {system_tokens} tokens")
    print(f"  Avg post content   : {avg_post_tokens:.0f} tokens  "
          f"(min={min(post_token_counts)}, max={max(post_token_counts)})")
    print(f"  Avg input / call   : {avg_input_per_call:.0f} tokens")
    print(f"  Est. output / call : {EST_OUTPUT_TOKENS} tokens")
    print(f"  Total input/model  : {total_input_per_model:,} tokens  ({total_input_per_model/1e6:.4f} MTok)")
    print(f"  Total output/model : {n * EST_OUTPUT_TOKENS:,} tokens  ({n * EST_OUTPUT_TOKENS/1e6:.4f} MTok)")
    print()

    col = 46
    print(f"{'Model':<{col}} {'Input $':>9} {'Output $':>9} {'Total $':>9}")
    print("-" * (col + 30))

    grand_total = 0.0
    for model, (in_price, out_price) in PRICING.items():
        input_cost  = (total_input_per_model / 1_000_000) * in_price
        output_cost = (n * EST_OUTPUT_TOKENS  / 1_000_000) * out_price
        total_cost  = input_cost + output_cost
        grand_total += total_cost
        print(f"{model:<{col}} ${input_cost:>8.4f} ${output_cost:>8.4f} ${total_cost:>8.4f}")

    print("-" * (col + 30))
    print(f"{'GRAND TOTAL':<{col}} {'':>9} {'':>9} ${grand_total:>8.4f}")
    print()
    print("Note: output token estimate is conservative. Actual reasoning length varies by model.")


if __name__ == "__main__":
    main()
