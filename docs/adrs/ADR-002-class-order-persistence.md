# ADR-002: Persisting Class Ordering in Checkpoints

## Status
Accepted

## Context
`torchvision.datasets.ImageFolder` assigns class indices alphabetically based on directory names. A previous mismatch between hardcoded class ordering and stored model logits caused systematic label permutation at inference.

## Decision
Store the dataset's resolved class list (`ImageFolder.classes`) inside each checkpoint under the key `classes` and load it dynamically during inference, overriding the default constant ordering.

## Rationale
- Eliminates dependency on assumed fixed ordering.
- Prevents silent semantic misalignment if directory names change or are subset.
- Backward-compatible: if `classes` absent, fallback to default tuple.

## Consequences
- Checkpoint consumers must respect stored class list (documented in README).
- Slightly larger checkpoint (negligible overhead).

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| Hardcode ordering | Simplicity | Fragile; mismatch risk |
| External JSON metadata | Explicit | Additional file management |
| Enum-based class registry | Type safety | Overkill for current scope |

## Future Evolution
- Add checksum or hash to detect drift between expected and loaded classes.
- Version `classes` schema if multi-label or hierarchical taxonomy introduced.
