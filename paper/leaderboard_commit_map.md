# Leaderboard commit-hash → owner map (Team SDG, PhysioNet Challenge 2026)

Resolves the automated-leaderboard snapshot (keyed by **git commit SHA-1**, scored by **AUROC**) to
identify which rows are ours. Each SHA was matched against the HEADs of our repos via `git ls-remote`
(verified, not guessed). **This is an AUROC-only snapshot** — see the caveat below.

Source: leaderboard paste — verbatim header `hash    auroc`, 6 rows.

## Rows, resolved (sorted by AUROC ↓)

| Commit (full SHA-1) | AUROC | Reward | Repo / owner | Whose |
|---|:---:|:---:|---|---|
| `244ca99f57037929116efba659fd48ad1d63e357` | **0.621** | n/a | `physionet26` / 4deg-kelvin | Team SDG (teammate) |
| `81035afb2beda3c3bc2468ab849cbcc363c9d80b` | **0.604** | n/a | `physionet26` / 4deg-kelvin | Team SDG (teammate) |
| `ba3f5d472bab9602ba7a280c72e51757eb7263ad` | **0.570** | n/a | `sleeperagents` / ArshiaIlaty | **Ours (Arshia)** |
| `f63391049ffa2e51fa1952cbb4518c294c11ddb9` | **0.545** | n/a | `physionet26` / 4deg-kelvin | Team SDG (teammate) |
| `9ca5d675b4ba87d02ce4f04272fa58484646aed6` | **0.533** | n/a | — (not in any of our repos) | **Another team** |
| `9857c25ee09acf483c2a9dd10a4ad21500432681` | **0.513** | n/a | `sleeperagents-core` / ArshiaIlaty | **Ours (Arshia)** |

## Answer to the original question ("which of these scores are ours?")

- **5 of 6 rows are Team SDG's**; `9ca5d675` (0.533) is another team's.
- **Arshia's two personal submissions:** `ba3f5d47` = **AUROC 0.570** (`sleeperagents`) and
  `9857c25e` = **AUROC 0.513** (`sleeperagents-core`).
- The three highest (0.621 / 0.604 / 0.545) are teammate 4deg-kelvin's `physionet26` pipeline.

## Caveat — Reward is NOT available for these rows, and the scales differ

- This leaderboard reported **AUROC only** (plain ranking); the paste had no reward column, and no reward
  value is tied to any of these 6 commit SHAs anywhere in our records. The `Reward` column above is
  therefore **n/a** — do not fill it with numbers from the other table.
- The **Reward / AC-AUROC** figures live in the separate **submission-ID progression** (2027…2872), e.g.
  official entry **2634: AC-AUROC 0.748 / Reward 0.168** (see `POSTER.md`, "Table 1 — Submission
  progression"). Those are AC-AUROC + Reward on a different snapshot and a different key (submission ID,
  not commit SHA), so the two tables must **not** be merged (0.570 AUROC here ≠ 0.748 AC-AUROC there).
- For a poster, headline the **Reward / AC-AUROC** table (Reward is the Challenge's primary metric); keep
  this commit-hash map as an internal "which rows are ours" reference.
