# merged_fpl_understat_all.csv column glossary

## FPL core match stats
- name: Player name (FPL).
- position: FPL position code (GK/DEF/MID/FWD).
- team: Player team name (FPL).
- xP: Expected FPL points (FPL model).
- assists: Assists in the match (FPL).
- bonus: Bonus points awarded (FPL).
- bps: Bonus points system score (FPL).
- clean_sheets: Clean sheets (FPL).
- creativity: Creativity index (FPL).
- element: FPL player ID.
- expected_assists: Expected assists (FPL model).
- expected_goal_involvements: Expected goal involvements (FPL model).
- expected_goals: Expected goals (FPL model).
- expected_goals_conceded: Expected goals conceded (FPL model).
- fixture: FPL fixture ID.
- goals_conceded: Goals conceded by player team (FPL).
- goals_scored: Goals scored by player (FPL).
- ict_index: ICT index (FPL).
- influence: Influence index (FPL).
- kickoff_time: Match kickoff timestamp (UTC).
- minutes: Minutes played.
- opponent_team: FPL opponent team ID.
- own_goals: Own goals.
- penalties_missed: Penalties missed.
- penalties_saved: Penalties saved (GK).
- red_cards: Red cards.
- round: Gameweek number (FPL).
- saves: Saves (GK).
- selected: Number of FPL teams selecting the player.
- starts: Matches started.
- team_a_score: Away team score (FPL).
- team_h_score: Home team score (FPL).
- threat: Threat index (FPL).
- total_points: FPL points earned in match.
- transfers_balance: Net transfers in minus out.
- transfers_in: Transfers in.
- transfers_out: Transfers out.
- value: Player value (FPL, in tenths).
- was_home: True if player team was home.
- yellow_cards: Yellow cards.
- GW: Gameweek (FPL).
- season: Season label (e.g., 2025-26).

## Opponent (from teams.csv)
- opponent_name: Opponent team name.
- opponent_short_name: Opponent team short code.
- opponent_code: Opponent team code.
- opponent_strength: Overall opponent strength.
- opponent_strength_overall_home: Opponent overall strength at home.
- opponent_strength_overall_away: Opponent overall strength away.
- opponent_strength_attack_home: Opponent attack strength at home.
- opponent_strength_attack_away: Opponent attack strength away.
- opponent_strength_defence_home: Opponent defence strength at home.
- opponent_strength_defence_away: Opponent defence strength away.

## Understat player match data
- us_goals: Goals (Understat).
- us_shots: Shots.
- us_xG: Expected goals (Understat).
- us_time: Minutes played (Understat).
- us_position: Position code (Understat).
- us_h_team: Home team name (Understat).
- us_a_team: Away team name (Understat).
- us_h_goals: Home goals (Understat match).
- us_a_goals: Away goals (Understat match).
- us_date: Match date (Understat).
- us_id: Understat match id.
- us_season: Understat season id.
- us_roster_id: Understat roster id.
- us_xA: Expected assists (Understat).
- us_assists: Assists (Understat).
- us_key_passes: Key passes.
- us_npg: Non-penalty goals.
- us_npxG: Non-penalty xG.
- us_xGChain: xGChain.
- us_xGBuildup: xGBuildup.
- us_isResult: Understat result flag.
- us_side: Understat side indicator.
- us_h: Understat home team id.
- us_a: Understat away team id.
- us_datetime: Understat match datetime.
- us_forecast: Understat forecast.

## IDs
- player_id: Understat player ID used for cross-season linking.
