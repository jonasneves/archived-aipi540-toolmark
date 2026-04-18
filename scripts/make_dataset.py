"""Build the Toolmark dataset from InjecAgent.

For each of the 1,054 InjecAgent attack cases, we emit two rows:
    - one **malicious** row using the case's `Tool Response` verbatim
    - one **benign** row using the case's `Tool Response Template` with the
      `<Attacker Instruction>` slot filled with a natural-sounding filler,
      drawn deterministically from a fixed pool

This produces ~2,108 balanced samples across 17 user tools / 8 tool classes,
with matched (benign, malicious) pairs anchored on the same template shell.

Scope note: BIPIA was considered but its schema is `context + question +
answer` (task-benchmark style, not labeled injection spans), so extracting
clean binary labels requires span-level parsing that exceeds the 4-day
budget. WASP requires a live web-agent sandbox and is similarly out of
scope. Leave-One-Tool-Class-Out on InjecAgent is the distribution-shift
experiment in place of cross-dataset transfer.

Label-noise hazard (documented in the report): benign fillers come from a
fixed pool of 30 natural phrases. A classifier may learn spurious signals
from the filler distribution. The report's error-analysis section inspects
this explicitly.

Sources:
    InjecAgent (ACL 2024 Findings) — Zhan et al.
    https://github.com/uiuc-kang-lab/InjecAgent
    https://aclanthology.org/2024.findings-acl.624/
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

INJECAGENT_FILES = {
    "dh_base": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
    "ds_base": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/test_cases_ds_base.json",
}

# Unified tool-class taxonomy. Each InjecAgent User Tool maps to exactly one
# class. Classes are the unit of the Leave-One-Tool-Class-Out experiment.
TOOL_CLASS_BY_USER_TOOL: dict[str, str] = {
    "AmazonGetProductDetails": "shopping",
    "ShopifyGetProductDetails": "shopping",
    "EvernoteManagerSearchNotes": "note",
    "TodoistSearchTasks": "note",
    "GitHubGetRepositoryDetails": "code",
    "GitHubGetUserDetails": "code",
    "GitHubSearchRepositories": "code",
    "GmailReadEmail": "email",
    "GmailSearchEmails": "email",
    "GoogleCalendarGetEventsFromSharedCalendar": "calendar",
    "GoogleCalendarReadEvents": "calendar",
    "TeladocViewReviews": "healthcare",
    "TwilioGetReceivedSmsMessages": "messaging",
    "TwitterManagerGetUserProfile": "messaging",
    "TwitterManagerReadTweet": "messaging",
    "TwitterManagerSearchTweets": "messaging",
    "WebBrowserNavigateTo": "web_fetch",
}

# Benign filler pool — 30 natural phrases spanning short / medium / longer
# forms. Deterministic per-case assignment (hash-indexed) so builds are
# reproducible. Chosen to sound plausibly like tool-output content without
# any instruction-giving phrasing.
BENIGN_FILLERS: tuple[str, ...] = (
    "Great product, fast shipping, would buy again.",
    "Thanks for the quick reply — see you Thursday.",
    "Meeting moved to conference room B, 3pm.",
    "All tests passed on the latest commit.",
    "Battery life has been excellent so far.",
    "Let me know if you need the draft earlier.",
    "Ticket resolved; closing this thread.",
    "Documentation lives in the wiki under onboarding.",
    "Looks good to me, approved.",
    "Running a bit late, will be there in 10.",
    "Flight is confirmed for the 14th, leaving at 6am.",
    "I picked up the dry cleaning on the way home.",
    "Reviewer requested minor revisions, nothing structural.",
    "Weather looks clear for the weekend trip.",
    "Groceries delivered — I put them on the counter.",
    "The article is short but well-written.",
    "Happy with the upgrade, performance noticeably better.",
    "Filed under project notes for later reference.",
    "Arrived on time, smooth flight, no delays.",
    "Check the appendix for the full error breakdown.",
    "Dinner is at seven, casual dress.",
    "Reminder: dentist appointment on the 22nd.",
    "Sample arrived in good condition, no packaging damage.",
    "Draft uploaded to the shared folder this morning.",
    "Call me when you're free, no rush.",
    "Thanks for the recommendation, exactly what I needed.",
    "Updated the spreadsheet with the final figures.",
    "The conference hotel is walking distance to the venue.",
    "Follow-up is scheduled for next Tuesday.",
    "Picked up a few things on sale, worth the detour.",
)


@dataclass(frozen=True)
class Record:
    """Canonical Toolmark row. Two emitted per InjecAgent case (benign + malicious)."""

    text: str
    label: int  # 0 benign, 1 malicious
    tool_class: str
    tool_name: str
    source: str
    template_id: str
    case_id: str
    attack_type: str | None
    variant: str  # "benign" or "malicious"


def fetch_injecagent(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download InjecAgent base test cases into raw_dir. Idempotent."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, url in INJECAGENT_FILES.items():
        dest = raw_dir / f"injecagent_{name}.json"
        if not dest.exists():
            log.info("fetching %s -> %s", url, dest)
            urllib.request.urlretrieve(url, dest)
        paths[name] = dest
    return paths


def _template_id(user_tool: str, attacker_instruction: str) -> str:
    """Stable hash of (tool, attack). Identical attacks across tools share ids."""
    digest = hashlib.sha256(attacker_instruction.encode("utf-8")).hexdigest()[:12]
    return f"{user_tool}:{digest}"


def _case_id(split: str, idx: int) -> str:
    return f"injecagent_{split}_{idx:04d}"


def _benign_filler_for(case_id: str) -> str:
    """Deterministic filler selection indexed by the case id."""
    h = int(hashlib.md5(case_id.encode("utf-8")).hexdigest(), 16)
    return BENIGN_FILLERS[h % len(BENIGN_FILLERS)]


def _synthesize_benign_text(template: str, filler: str) -> str:
    return template.replace("<Attacker Instruction>", filler)


def _normalize_response(text: str) -> str:
    """Strip the outer JSON-string quotes InjecAgent wraps `Tool Response` in.

    The source stores responses as JSON-encoded strings, which round-trip
    through our parser as literal `"..."` wrappers. `Tool Response Template`
    has no such wrapper, so leaving them in creates a spurious signal the
    classifier can latch onto (outer-quote presence = malicious).
    """
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1]
    return stripped


def _iter_records(raw_paths: dict[str, Path]) -> Iterator[Record]:
    """Yield matched (benign, malicious) records per InjecAgent case."""
    for split_name, path in raw_paths.items():
        with path.open() as f:
            cases = json.load(f)
        for idx, row in enumerate(cases):
            user_tool = row["User Tool"]
            tool_class = TOOL_CLASS_BY_USER_TOOL.get(user_tool)
            if tool_class is None:
                log.warning("unknown user tool, skipping: %s", user_tool)
                continue
            case_id = _case_id(split_name, idx)
            tmpl_id = _template_id(user_tool, row["Attacker Instruction"])
            attack_type = row.get("Attack Type")

            yield Record(
                text=_normalize_response(row["Tool Response"]),
                label=1,
                tool_class=tool_class,
                tool_name=user_tool,
                source="injecagent",
                template_id=tmpl_id,
                case_id=case_id,
                attack_type=attack_type,
                variant="malicious",
            )

            filler = _benign_filler_for(case_id)
            yield Record(
                text=_synthesize_benign_text(row["Tool Response Template"], filler),
                label=0,
                tool_class=tool_class,
                tool_name=user_tool,
                source="injecagent",
                template_id=tmpl_id,
                case_id=case_id,
                attack_type=None,
                variant="benign",
            )


def build_dataframe(raw_paths: dict[str, Path]) -> pd.DataFrame:
    rows = [record.__dict__ for record in _iter_records(raw_paths)]
    return pd.DataFrame(rows)


def dedup_by_template(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence per (template_id, variant).

    Attack templates can repeat across the dh/ds splits. Deduplication
    matters for honest evaluation — reported per SCOPE.md.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["template_id", "variant"], keep="first").reset_index(drop=True)
    log.info("dedup: %d -> %d rows", before, len(df))
    return df


def summarize(df: pd.DataFrame) -> dict[str, object]:
    return {
        "total_rows": int(len(df)),
        "by_label": df["label"].value_counts().to_dict(),
        "by_variant": df["variant"].value_counts().to_dict(),
        "by_tool_class": df["tool_class"].value_counts().to_dict(),
        "by_tool_name": df["tool_name"].value_counts().to_dict(),
        "unique_templates": int(df["template_id"].nunique()),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_paths = fetch_injecagent()
    df = build_dataframe(raw_paths)
    df = dedup_by_template(df)

    out_path = PROCESSED_DIR / "toolmark.parquet"
    df.to_parquet(out_path, index=False)
    log.info("wrote %s (%d rows)", out_path, len(df))

    summary = summarize(df)
    summary_path = PROCESSED_DIR / "toolmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, default=int)
    log.info("wrote %s", summary_path)


if __name__ == "__main__":
    main()
