"""Fetch InjecAgent, BIPIA, and WASP; join with tool metadata preserved.

Writes data/processed/toolmark.parquet with columns:
    text           raw tool output / attack payload
    label          0 = benign, 1 = injection
    tool_class     normalized tool category (shell, web_fetch, file_read,
                   email, mcp_metadata, other)
    tool_name      original tool identifier
    source         injecagent | bipia | wasp
    template_id    stable hash of the attack template (for dedup)

TODO(Sat): implement fetchers per dataset with attack-template dedup.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("implemented on Sat Apr 18; see SCOPE.md timeline")


if __name__ == "__main__":
    main()
