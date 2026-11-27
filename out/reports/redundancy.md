# Redundancy report

Building: **B1**

## Single-failure sweep

- 283 elements tested (doors, fire doors, exits, openings, stairs, corridor segments, gates, walkway segments).
- Rooms disconnected by *some* single failure: FF:F-01, FF:F-04, FF:F-05, FF:F-07, FF:F-09, FF:F-11, FF:F-12, FF:F-14, FF:F-15, FF:F-17, FF:F-20, GF:G-01, GF:G-02, GF:G-03, GF:G-04, GF:G-06, GF:G-08, GF:G-09, GF:G-10, GF:G-11, GF:G-12, GF:G-13, GF:G-14, GF:G-15, GF:G-17, GF:G-18, GF:G-19, GF:G-20, GF:G-22, GF:G-23, GF:G-24, GF:G-26, GF:G-28, GF:G-30, GF:G-31
- Acceptance rule (configs/agent_defaults.yaml `redundancy:`, generic -- no room id ever named here): a disconnected room is acceptable iff it has exactly 1 door AND occupancy <= 49 AND common path of travel <= 23 m.
- Declared `designed_dead_ends` (reference only, not the acceptance rule): FF:F-01, FF:F-04, FF:F-05, FF:F-07, FF:F-09, FF:F-11, FF:F-12, FF:F-14, FF:F-15, FF:F-17, FF:F-20, GF:G-01, GF:G-02, GF:G-03, GF:G-04, GF:G-06, GF:G-07, GF:G-08, GF:G-09, GF:G-10, GF:G-11, GF:G-12, GF:G-14, GF:G-15, GF:G-17, GF:G-18, GF:G-19, GF:G-20, GF:G-22, GF:G-23, GF:G-24, GF:G-26, GF:G-28, GF:G-30, GF:G-31
- Every disconnected room passes the rule: **NO -- see unexpected rows below**

### Disconnected-room detail

| Room | Use | Max occupancy | Doors | Common path (m) | Acceptable |
|---|---|---|---|---|---|
| FF:F-01 | office | 24 | 1 | 9.7 | yes |
| FF:F-04 | office | 19 | 1 | 9.3 | yes |
| FF:F-05 | office | 14 | 1 | 9.9 | yes |
| FF:F-07 | office | 6 | 1 | 6.9 | yes |
| FF:F-09 | hvac_plant | 1 | 1 | 8.2 | yes |
| FF:F-11 | office | 22 | 1 | 9.7 | yes |
| FF:F-12 | library | 5 | 1 | 10.5 | yes |
| FF:F-14 | store | 1 | 1 | 4.0 | yes |
| FF:F-15 | office | 25 | 1 | 10.9 | yes |
| FF:F-17 | toilets | 3 | 1 | 6.3 | yes |
| FF:F-20 | office | 6 | 1 | 6.2 | yes |
| GF:G-01 | workshop | 6 | 1 | 10.9 | yes |
| GF:G-02 | office | 12 | 1 | 9.0 | yes |
| GF:G-03 | office | 10 | 1 | 8.8 | yes |
| GF:G-04 | office | 10 | 1 | 8.8 | yes |
| GF:G-06 | office | 14 | 1 | 8.5 | yes |
| GF:G-08 | office | 6 | 1 | 6.2 | yes |
| GF:G-09 | toilets | 3 | 1 | 7.6 | yes |
| GF:G-10 | office | 6 | 1 | 6.9 | yes |
| GF:G-11 | office | 7 | 1 | 7.9 | yes |
| GF:G-12 | plant_room | 1 | 1 | 8.2 | yes |
| GF:G-13 | control_room | 8 | 2 | 10.9 | **NO** |
| GF:G-14 | meeting | 39 | 1 | 10.3 | yes |
| GF:G-15 | office | 16 | 1 | 9.4 | yes |
| GF:G-17 | store | 1 | 1 | 11.0 | yes |
| GF:G-18 | office | 19 | 2 | 9.9 | **NO** |
| GF:G-19 | store | 1 | 1 | 21.3 | yes |
| GF:G-20 | office | 25 | 1 | 10.9 | yes |
| GF:G-22 | office | 9 | 1 | 7.9 | yes |
| GF:G-23 | office | 9 | 1 | 7.5 | yes |
| GF:G-24 | store | 1 | 1 | 6.3 | yes |
| GF:G-26 | office | 8 | 1 | 8.4 | yes |
| GF:G-28 | meeting | 14 | 1 | 6.2 | yes |
| GF:G-30 | plant_room | 1 | 1 | 7.9 | yes |
| GF:G-31 | office | 8 | 1 | 5.8 | yes |

### Unexpected disconnections (fail the generic single-door-room rule)

| Failed element | Kind | Rooms disconnected |
|---|---|---|
| GF:G-18-d1 | door | GF:G-18 |
| GF:C:C-N-01 | corridor_segment | GF:G-13 |
| GF:C:C-N-13 | corridor_segment | GF:G-18 |

### Reference-room cost impact (worst 10 by cost increase)

| Failed element | Kind | G-13 | F-10 |
|---|---|---|---|
| GF:C:C-N-01 | corridor_segment | **UNREACHABLE** | 75.7s (+0%) |
| FF:ST-W-d1 | fire_door | 64.3s (+0%) | 122.1s (+61%) |
| ST-W | stair | 64.3s (+0%) | 122.1s (+61%) |
| FF:C:C-W-02 | corridor_segment | 64.3s (+0%) | 122.1s (+61%) |
| FF:open:C-N|C-W:6 | opening | 64.3s (+0%) | 112.9s (+49%) |
| FF:C:C-N-00 | corridor_segment | 64.3s (+0%) | 112.9s (+49%) |
| FF:C:C-N-01 | corridor_segment | 64.3s (+0%) | 112.9s (+49%) |
| FF:C:C-W-03 | corridor_segment | 64.3s (+0%) | 112.9s (+49%) |
| FF:C:C-W-04 | corridor_segment | 64.3s (+0%) | 112.9s (+49%) |
| GF:open:C-N|C-W:7 | opening | 91.2s (+42%) | 75.7s (+0%) |

## Double-failure scenarios

### two_stairs_lost

- Rooms disconnected: none
- GF:R:G-13: 64.3s (+0%)
- FF:R:F-10: 75.7s (+0%)

### Every exit pair

- 21 pairs tested.
- Pairs that disconnect at least one room: 0

## FF ring edge-disjoint paths to an assembly point

| Ring segment node | Edge-disjoint paths found (cutoff=2) |
|---|---|

