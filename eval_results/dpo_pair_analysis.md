# DPO Pair Distribution Analysis

**Total pairs**: 2377
**Unique problems**: 534

## Solution Length (characters)
| | Chosen | Rejected | Δ |
|---|---|---|---|
| Mean | 16124 | 21128 | -5004 |
| Median | 14148 | 18315 | -4167 |
| P10 | 5240 | 6476 | |
| P90 | 29580 | 39518 | |

## Length Ratio (chosen/rejected)
- **Chosen shorter (<0.9×)**: 1408 (59.2%)
- **Similar length (0.9-1.1×)**: 327 (13.8%)
- **Chosen longer (>1.1×)**: 642 (27.0%)
- **Mean ratio**: 0.90

This confirms StructPO is NOT a length-based preference — it's structural.


## Rollout Statistics
| Metric | Value |
|---|---|
| Total problems | 817 |
| Total traces | 6536 |
| Correct traces | 5097 (78.0%) |
| Incorrect traces | 1439 (22.0%) |
| Avg DSR (all) | 0.223 |
| Avg DSR (correct) | 0.188 |
| Avg DSR (incorrect) | 0.350 |
| Avg steps (all) | 199.4 |
| Avg trace len (chars) | 19492 |

## DSR Distribution (all traces)
| DSR range | Count | % |
|---|---|---|
| 0.00 | 744 | 11.4% |
| 0.01-0.09 | 3538 | 54.1% |
| 0.10-0.19 | 402 | 6.2% |
| 0.20-0.29 | 161 | 2.5% |
| 0.30-0.39 | 143 | 2.2% |
| 0.40-0.49 | 140 | 2.1% |
| 0.50+ | 1408 | 21.5% |

## Pair Type Breakdown
| Type | Count | % | Description |
|---|---|---|---|
| **Efficiency** | 1074 | 45.2% | correct low-DSR > correct high-DSR |
| **Productive Exploration** | 790 | 33.2% | live verification > dead verification |
| **Direction** | 513 | 21.6% | correct efficient > incorrect wasteful |
| **Total** | 2377 | 100% | |

## Problem Coverage
| Type | Problems | Pairs/Problem |
|---|---|---|
| Efficiency | 368 | 2.9 |
| Productive Exploration | 307 | 2.6 |
| Direction | 192 | 2.7 |
| **Any type** | **534** / 817 | 4.5 |

## DSR Gap Analysis
| Type | Chosen DSR (mean) | Rejected DSR (mean) | Gap |
|---|---|---|---|
| Efficiency | 0.047 | 0.723 | 0.675 |
| Productive Expl. | 0.050 | 0.470 | 0.421 |
| Direction | 0.065 | 0.733 | 0.668 |

## Correctness in Pairs
| Type | Both correct | Chosen correct only |
|---|---|---|
| Efficiency | 1074 (100%) | 0 (0%) |
| Productive Expl. | 790 (100%) | 0 (0%) |
| Direction | 0 (0%) | 513 (100%) |
