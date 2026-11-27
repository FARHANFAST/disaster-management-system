# Redundancy report

Building: **B2**

## Single-failure sweep

- 100 elements tested (doors, fire doors, exits, openings, stairs, corridor segments, gates, walkway segments).
- Rooms disconnected by *some* single failure: L1:N1b, L1:S1b, L1:S3a, L1:S3b, L1:S5a, L2:N1, L2:N1b, L2:N2, L2:S1a, L2:S1b, L2:S2, L2:S3a, L2:S3b, L2:S4, L2:S5a, L2:S5b, L3:L3-M1, L3:L3-M2, L3:N1, L3:N1b, L3:N2, L3:S1a, L3:S1b, L3:S2, L3:S3a, L3:S3b, L3:S4, L3:S5a, L3:S5b
- Acceptance rule (configs/agent_defaults.yaml `redundancy:`, generic -- no room id ever named here): a disconnected room is acceptable iff it has exactly 1 door AND occupancy <= 49 AND common path of travel <= 23 m.
- Declared `designed_dead_ends` (reference only, not the acceptance rule): 
- Every disconnected room passes the rule: **YES**

### Disconnected-room detail

| Room | Use | Max occupancy | Doors | Common path (m) | Acceptable |
|---|---|---|---|---|---|
| L1:N1b | office | 0 | 1 | 4.9 | yes |
| L1:S1b | office | 0 | 1 | 6.5 | yes |
| L1:S3a | office | 0 | 1 | 7.2 | yes |
| L1:S3b | office | 0 | 1 | 7.1 | yes |
| L1:S5a | office | 0 | 1 | 7.2 | yes |
| L2:N1 | office | 0 | 1 | 17.4 | yes |
| L2:N1b | office | 0 | 1 | 4.9 | yes |
| L2:N2 | plant_room | 0 | 1 | 7.9 | yes |
| L2:S1a | office | 0 | 1 | 9.9 | yes |
| L2:S1b | office | 0 | 1 | 6.5 | yes |
| L2:S2 | lobby | 0 | 1 | 6.9 | yes |
| L2:S3a | office | 0 | 1 | 7.2 | yes |
| L2:S3b | office | 0 | 1 | 7.1 | yes |
| L2:S4 | plant_room | 0 | 1 | 7.1 | yes |
| L2:S5a | office | 0 | 1 | 7.2 | yes |
| L2:S5b | office | 0 | 1 | 9.7 | yes |
| L3:L3-M1 | office | 0 | 1 | 8.1 | yes |
| L3:L3-M2 | office | 0 | 1 | 8.5 | yes |
| L3:N1 | office | 0 | 1 | 17.4 | yes |
| L3:N1b | office | 0 | 1 | 4.9 | yes |
| L3:N2 | office | 0 | 1 | 7.9 | yes |
| L3:S1a | office | 0 | 1 | 9.9 | yes |
| L3:S1b | office | 0 | 1 | 6.5 | yes |
| L3:S2 | lobby | 0 | 1 | 6.9 | yes |
| L3:S3a | office | 0 | 1 | 6.2 | yes |
| L3:S3b | office | 0 | 1 | 6.4 | yes |
| L3:S4 | plant_room | 0 | 1 | 7.1 | yes |
| L3:S5a | office | 0 | 1 | 7.2 | yes |
| L3:S5b | office | 0 | 1 | 9.7 | yes |

### Reference-room cost impact (worst 10 by cost increase)

| Failed element | Kind | S2 | S2 | S2 |
|---|---|---|---|---|
| S2-d1 | door | 41.6s (+0%) | 84.0s (+0%) | **UNREACHABLE** |
| L2:C:corr-A-04 | corridor_segment | 41.6s (+0%) | **UNREACHABLE** | 95.7s (+0%) |
| L3:C:corr-A-04 | corridor_segment | 41.6s (+0%) | 84.0s (+0%) | **UNREACHABLE** |
| X-S | exit | 66.5s (+60%) | 88.2s (+5%) | 98.4s (+3%) |
| GATE3-S | gate | 47.3s (+14%) | 88.2s (+5%) | 98.4s (+3%) |
| L1:C:corr-A-04 | corridor_segment | 41.6s (+0%) | 88.2s (+5%) | 98.4s (+3%) |
| L1:C:corr-A-05 | corridor_segment | 41.6s (+0%) | 88.2s (+5%) | 98.4s (+3%) |
| L2:C:corr-A-05 | corridor_segment | 41.6s (+0%) | 88.2s (+5%) | 95.7s (+0%) |
| ST-C-op-corrA | opening | 41.6s (+0%) | 84.0s (+0%) | 98.4s (+3%) |
| L3:C:corr-A-05 | corridor_segment | 41.6s (+0%) | 84.0s (+0%) | 98.4s (+3%) |

## Double-failure scenarios

### Every exit pair

- 28 pairs tested.
- Pairs that disconnect at least one room: 0

## FF ring edge-disjoint paths to an assembly point

| Ring segment node | Edge-disjoint paths found (cutoff=2) |
|---|---|

