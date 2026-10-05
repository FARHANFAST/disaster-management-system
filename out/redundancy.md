# Redundancy report

Building: **B1**

## Single-failure sweep

- 163 elements tested (doors, fire doors, exits, openings, stairs, corridor segments).
- Rooms disconnected by *some* single failure: F-01, F-04, F-08, F-09, F-10, F-13, G-01, G-03, G-04, G-07, G-08, G-09, G-10, G-12, G-13, G-14, G-17, G-18, G-20
- Declared `designed_dead_ends`: F-01, F-04, F-08, F-09, F-10, F-13, G-01, G-03, G-04, G-07, G-08, G-09, G-10, G-12, G-13, G-14, G-17, G-18, G-20
- Matches declared set exactly: **YES**

### Reference-room cost impact (worst 10 by cost increase)

| Failed element | Kind | G-16 | G-11 | F-07 |
|---|---|---|---|---|
| ST-W-fd-FF | fire_door | 45.0s (+0%) | 45.0s (+0%) | 89.5s (+58%) |
| ST-W | stair | 45.0s (+0%) | 45.0s (+0%) | 89.5s (+58%) |
| FF:C:ring-W-05 | corridor_segment | 45.0s (+0%) | 45.0s (+0%) | 89.5s (+58%) |
| FF:C:ring-W-06 | corridor_segment | 45.0s (+0%) | 45.0s (+0%) | 89.5s (+58%) |
| GF:C:ring-W-06 | corridor_segment | 45.0s (+0%) | 61.6s (+37%) | 56.5s (+0%) |
| GF:C:ring-W-07 | corridor_segment | 45.0s (+0%) | 61.6s (+37%) | 56.5s (+0%) |
| X-NW | exit | 45.0s (+0%) | 51.5s (+14%) | 72.7s (+29%) |
| F-07-d1 | door | 45.0s (+0%) | 45.0s (+0%) | 70.9s (+26%) |
| G-11-d1 | door | 45.0s (+0%) | 53.2s (+18%) | 56.5s (+0%) |
| G-16-d2 | door | 52.2s (+16%) | 45.0s (+0%) | 56.5s (+0%) |

## Double-failure scenarios

### two_stairs_lost (ST-C + ST-E)

- Rooms disconnected: none
- GF:R:G-16: 45.0s (+0%)
- GF:R:G-11: 45.0s (+0%)
- FF:R:F-07: 56.5s (+0%)

### block_main_exit + block_ringS_mid

- Rooms disconnected: none
- GF:R:G-16: 45.0s (+0%)
- GF:R:G-11: 45.0s (+0%)
- FF:R:F-07: 56.5s (+0%)

### Every exit pair

- 15 pairs tested.
- Pairs that disconnect at least one room: 0

## FF ring edge-disjoint paths to an assembly point

| Ring segment node | Edge-disjoint paths found (cutoff=2) |
|---|---|
| FF:C:ring-E-00 | 2 |
| FF:C:ring-E-01 | 2 |
| FF:C:ring-E-02 | 2 |
| FF:C:ring-E-03 | 2 |
| FF:C:ring-E-04 | 2 |
| FF:C:ring-E-05 | 2 |
| FF:C:ring-N-00 | 2 |
| FF:C:ring-N-01 | 2 |
| FF:C:ring-N-02 | 2 |
| FF:C:ring-N-03 | 2 |
| FF:C:ring-N-04 | 2 |
| FF:C:ring-N-05 | 2 |
| FF:C:ring-N-06 | 2 |
| FF:C:ring-N-07 | 2 |
| FF:C:ring-N-08 | 2 |
| FF:C:ring-S-00 | 2 |
| FF:C:ring-S-01 | 2 |
| FF:C:ring-S-02 | 2 |
| FF:C:ring-S-03 | 2 |
| FF:C:ring-S-04 | 2 |
| FF:C:ring-S-05 | 2 |
| FF:C:ring-S-06 | 2 |
| FF:C:ring-S-07 | 2 |
| FF:C:ring-S-08 | 2 |
| FF:C:ring-S-09 | 2 |
| FF:C:ring-W-00 | 1  <-- BELOW TARGET |
| FF:C:ring-W-01 | 2 |
| FF:C:ring-W-02 | 2 |
| FF:C:ring-W-03 | 2 |
| FF:C:ring-W-04 | 2 |
| FF:C:ring-W-05 | 2 |
| FF:C:ring-W-06 | 2 |

Segments below target are not necessarily a redundancy defect: a ring segment whose *only* other neighbour is a single-door dead-end room has no second useful path by construction (a detour into that room just leads back out the same door). Verify each one before treating it as a finding:

- `FF:C:ring-W-00`

