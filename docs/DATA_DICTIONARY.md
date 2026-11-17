# Phase 1 data dictionary

| Field | Description | Unit / caveat |
|---|---|---|
| game_pk | Game identifier | MLB game key |
| game_date | Game date | Date |
| batter / pitcher | MLBAM player IDs | Role-specific identifiers |
| pitch_type | Pitch classification | Categorical |
| release_speed | Pitch velocity | mph |
| release_spin_rate | Spin rate | rpm |
| pfx_x / pfx_z | Movement components | Coordinate convention must be documented before interpretation |
| launch_speed | Exit velocity | mph |
| launch_angle | Launch angle | degrees |
| estimated_woba_using_speedangle | Expected wOBA on batted-ball events | Not identical to season-level xwOBA |
| events | Plate-appearance result | Usually populated on terminal pitch |
| description | Pitch result | Source text |

`raw_statcast` preserves the source columns. `fact_statcast_pitch` provides a typed subset. Add fields through documented schema changes and tests.
