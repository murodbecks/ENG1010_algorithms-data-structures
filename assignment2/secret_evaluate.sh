#!/bin/bash

# ==========================================
# CONFIGURATION
# ==========================================
MY_BOT="${1:-"Scout"}"
RUN_CMD="python runner.py"
MAX_TICKS=10000

# ==========================================
# PARSER FUNCTION
# ==========================================
parse_result() {
    local raw_output="$1"
    echo "$raw_output" | grep -A 10 "RANKINGS:" | awk -v bot="$MY_BOT" '$3 == bot {print $1, $4; exit}'
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
get_points() {
    local rank=$1
    local is_duel=$2

    if [[ "$is_duel" == "true" ]]; then
        if [[ "$rank" -eq 1 ]]; then echo 100; else echo 0; fi
    else
        case "$rank" in
            1) echo 100 ;;
            2) echo 50 ;;
            3) echo 25 ;;
            *) echo 0 ;;
        esac
    fi
}

run_match() {
    local map_name=$1
    local b1=$2
    local b2=$3
    local b3=$4
    local b4=$5
    local is_duel=$6

    local output=$($RUN_CMD -m "${map_name}.txt" -b "$b1" "$b2" "$b3" "$b4" --max-ticks $MAX_TICKS)
    local result=$(parse_result "$output")
    local rank=$(echo "$result" | awk '{print $1}')
    local game_score=$(echo "$result" | awk '{print $2}')

    if [[ -z "$rank" ]]; then
        rank=4
        game_score="N/A"
    fi

    local points=$(get_points "$rank" "$is_duel")

    echo -e "  -> Map: ${map_name}.txt | Bots: $b1, $b2, $b3, $b4" >&2
    echo -e "     [Result] Rank: $rank | Game Score: $game_score | Points: $points\n" >&2
    echo "$points"
}

# ==========================================
# MAIN EXECUTION
# ==========================================
echo "==========================================="
echo "   STARTING SECRET EVALUATION FOR: $MY_BOT"
echo "==========================================="

HIDDEN_SOLO_MAPS=("s_path_4" "s_floodfill_3" "s_choice_3")
SECRET_PVP_MAPS=("secret_cross" "secret_islands" "secret_choke")

# --- 1. HIDDEN SOLO PLAYER (8 Points) ---
echo -e "\n--- Running Hidden Solo Scenarios ---"
SOLO_SCORE=0
SOLO_MAX=300 # 3 maps * 100 points max

for map in "${HIDDEN_SOLO_MAPS[@]}"; do
    pts=$(run_match "$map" "$MY_BOT" "Drunk" "Drunk" "Drunk" "false")
    SOLO_SCORE=$((SOLO_SCORE + pts))
done

# --- 2. HIDDEN DUEL (10 Points) ---
echo -e "\n--- Running Hidden Duel Scenarios ---"
DUEL_BOTS=("Scout" "Rogue" "Stingray" "Viper" "Blaze")
DUEL_SCORE=0
DUEL_MAX=3000 # 3 maps * 5 bots * 2 positions * 100 points

for map in "${SECRET_PVP_MAPS[@]}"; do
    for opp in "${DUEL_BOTS[@]}"; do
        pts1=$(run_match "$map" "$MY_BOT" "Dummy" "Dummy" "$opp" "true")
        DUEL_SCORE=$((DUEL_SCORE + pts1))
        
        pts2=$(run_match "$map" "$opp" "Dummy" "Dummy" "$MY_BOT" "true")
        DUEL_SCORE=$((DUEL_SCORE + pts2))
    done
done

# --- 3. HIDDEN BATTLE ROYALE (7 Points) ---
echo -e "\n--- Running Hidden Battle Royale Scenarios ---"
BR_SCORE=0
BR_MAX=600 # 6 matches * 100 points max

while read -r map b1 b2 b3 b4; do
    [ -z "$map" ] && continue
    pts=$(run_match "$map" "$b1" "$b2" "$b3" "$b4" "false")
    BR_SCORE=$((BR_SCORE + pts))
done <<EOF
secret_cross $MY_BOT Scout Stingray Blaze
secret_cross $MY_BOT Rogue Viper Stingray
secret_islands $MY_BOT Rogue Stingray Stingray
secret_islands $MY_BOT Viper Viper Rogue
secret_choke $MY_BOT Rogue Rogue Scout
secret_choke $MY_BOT Viper Stingray Scout
EOF

# ==========================================
# FINAL CALCULATIONS
# ==========================================
echo "==========================================="
echo "         HIDDEN SCENARIO RESULTS           "
echo "==========================================="

SOLO_WEIGHTED=$(awk "BEGIN {printf \"%.2f\", ($SOLO_SCORE / $SOLO_MAX) * 8}")
DUEL_WEIGHTED=$(awk "BEGIN {printf \"%.2f\", ($DUEL_SCORE / $DUEL_MAX) * 10}")
BR_WEIGHTED=$(awk "BEGIN {printf \"%.2f\", ($BR_SCORE / $BR_MAX) * 7}")
TOTAL_WEIGHTED=$(awk "BEGIN {printf \"%.2f\", $SOLO_WEIGHTED + $DUEL_WEIGHTED + $BR_WEIGHTED}")

echo "Solo Score:          $SOLO_SCORE / $SOLO_MAX points ($SOLO_WEIGHTED / 8.00 wt)"
echo "Duel Score:          $DUEL_SCORE / $DUEL_MAX points ($DUEL_WEIGHTED / 10.00 wt)"
echo "Battle Royale Score: $BR_SCORE / $BR_MAX points ($BR_WEIGHTED / 7.00 wt)"
echo "-------------------------------------------"
echo "TOTAL HIDDEN SCORE:      $TOTAL_WEIGHTED / 25.00"
echo "==========================================="