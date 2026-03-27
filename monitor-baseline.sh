#!/bin/bash
# Monitor script for baseline auditor run
# Checks every 20 minutes, appends to monitor-baseline.log
# Stops automatically when all processes finish

BASE="/Users/hermine/Documents/Projects/GitHub/NCDIT-Cabinet-ADA-FILES"
LOG="$BASE/monitor-baseline.log"
AGENCIES="commerce deq dmva doa dpi it labor ncagr ncdcr ncdhhs ncdoi ncdor ncdps nctreasurer"
INTERVAL=1200  # 20 minutes

echo "=== Baseline Auditor Monitor Started at $(date) ===" >> "$LOG"
echo "Checking every 20 minutes. PID=$$" >> "$LOG"
echo "" >> "$LOG"

while true; do
    echo "================================================================" >> "$LOG"
    echo "CHECK at $(date)" >> "$LOG"
    echo "================================================================" >> "$LOG"

    # 1. Count *-audit-report-baseline.json files per agency
    echo "" >> "$LOG"
    echo "--- Baseline report counts per agency ---" >> "$LOG"
    TOTAL_REPORTS=0
    for agency in $AGENCIES; do
        DIR="$BASE/$agency/htmls"
        if [ -d "$DIR" ]; then
            COUNT=$(find "$DIR" -name "*-audit-report-baseline.json" 2>/dev/null | wc -l | tr -d ' ')
            TOTAL_DOCS=$(find "$DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
            echo "  $agency: $COUNT / $TOTAL_DOCS baseline reports" >> "$LOG"
            TOTAL_REPORTS=$((TOTAL_REPORTS + COUNT))
        fi
    done
    echo "  TOTAL: $TOTAL_REPORTS baseline reports" >> "$LOG"

    # 2. Check if auditor processes are still running
    echo "" >> "$LOG"
    echo "--- Running auditor processes ---" >> "$LOG"
    PROCS=$(ps aux | grep '[a]uditor.py' | grep -v monitor)
    if [ -n "$PROCS" ]; then
        PROC_COUNT=$(echo "$PROCS" | wc -l | tr -d ' ')
        echo "  $PROC_COUNT auditor process(es) running" >> "$LOG"
        echo "$PROCS" | awk '{print "    PID=" $2 " CPU=" $3 "% MEM=" $4 "% CMD=" substr($0, index($0,$11))}' >> "$LOG"
    else
        echo "  No auditor processes running" >> "$LOG"
    fi

    # Also check run_audit_multi_project.py
    RUNNER_PROCS=$(ps aux | grep '[r]un_audit_multi_project.py')
    if [ -n "$RUNNER_PROCS" ]; then
        RUNNER_COUNT=$(echo "$RUNNER_PROCS" | wc -l | tr -d ' ')
        echo "  $RUNNER_COUNT runner process(es) active" >> "$LOG"
    else
        echo "  No runner processes active" >> "$LOG"
    fi

    # 3. Parse audit-batch-results-baseline.json for completed agencies
    echo "" >> "$LOG"
    echo "--- Completed agency batch results ---" >> "$LOG"
    for agency in $AGENCIES; do
        RESULTS="$BASE/$agency/htmls/audit-batch-results-baseline.json"
        if [ -f "$RESULTS" ]; then
            TOTAL=$(python3 -c "
import json
with open('$RESULTS') as f:
    d = json.load(f)
results = d.get('results', [])
total = len(results)
ok = sum(1 for r in results if r.get('status') == 'ok')
err = sum(1 for r in results if r.get('status') == 'error')
timeout = sum(1 for r in results if 'timeout' in str(r.get('error','')).lower() or 'timed out' in str(r.get('error','')).lower())
print(f'  {\"$agency\"}: {total} docs — {ok} ok, {err} errors, {timeout} timeouts')
" 2>/dev/null)
            if [ -n "$TOTAL" ]; then
                echo "$TOTAL" >> "$LOG"
            else
                echo "  $agency: batch results exist but parse error" >> "$LOG"
            fi
        fi
    done

    # 4. Check recent baseline reports for quota errors
    echo "" >> "$LOG"
    echo "--- Quota error check (RESOURCE_EXHAUSTED / rateLimitExceeded) ---" >> "$LOG"
    QUOTA_HITS=0
    for agency in $AGENCIES; do
        DIR="$BASE/$agency/htmls"
        if [ -d "$DIR" ]; then
            # Check reports modified in the last 25 minutes for quota errors
            RECENT=$(find "$DIR" -name "*-audit-report-baseline.json" -mmin -25 2>/dev/null)
            if [ -n "$RECENT" ]; then
                for f in $RECENT; do
                    if grep -ql 'RESOURCE_EXHAUSTED\|rateLimitExceeded' "$f" 2>/dev/null; then
                        SLUG=$(basename "$f" | sed 's/-audit-report-baseline.json//')
                        echo "  QUOTA ERROR: $agency / $SLUG" >> "$LOG"
                        QUOTA_HITS=$((QUOTA_HITS + 1))
                    fi
                done
            fi
        fi
    done
    if [ "$QUOTA_HITS" -eq 0 ]; then
        echo "  No quota errors in recent reports" >> "$LOG"
    else
        echo "  $QUOTA_HITS report(s) with quota errors" >> "$LOG"
    fi

    # 5. Check for V8 hard vetoes (all LLM calls failed)
    echo "" >> "$LOG"
    echo "--- V8 hard vetoes (all LLM calls failed) ---" >> "$LOG"
    V8_COUNT=0
    for agency in $AGENCIES; do
        DIR="$BASE/$agency/htmls"
        if [ -d "$DIR" ]; then
            for f in $(find "$DIR" -name "*-audit-report-baseline.json" 2>/dev/null); do
                if grep -ql '"V8"' "$f" 2>/dev/null; then
                    SLUG=$(basename "$f" | sed 's/-audit-report-baseline.json//')
                    echo "  V8 VETO: $agency / $SLUG" >> "$LOG"
                    V8_COUNT=$((V8_COUNT + 1))
                fi
            done
        fi
    done
    if [ "$V8_COUNT" -eq 0 ]; then
        echo "  No V8 vetoes found" >> "$LOG"
    else
        echo "  $V8_COUNT total V8 veto(es)" >> "$LOG"
    fi

    echo "" >> "$LOG"

    # 6. Stop automatically when all processes finish
    ALL_PROCS=$(ps aux | grep -E '[a]uditor.py|[r]un_audit_multi_project.py' | grep -v monitor)
    if [ -z "$ALL_PROCS" ]; then
        echo "*** ALL AUDITOR PROCESSES HAVE FINISHED ***" >> "$LOG"
        echo "Monitor stopping at $(date)" >> "$LOG"

        # Final summary
        echo "" >> "$LOG"
        echo "=== FINAL SUMMARY ===" >> "$LOG"
        for agency in $AGENCIES; do
            DIR="$BASE/$agency/htmls"
            COUNT=$(find "$DIR" -name "*-audit-report-baseline.json" 2>/dev/null | wc -l | tr -d ' ')
            TOTAL_DOCS=$(find "$DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
            echo "  $agency: $COUNT / $TOTAL_DOCS" >> "$LOG"
        done
        GRAND_TOTAL=$(find "$BASE" -name "*-audit-report-baseline.json" 2>/dev/null | wc -l | tr -d ' ')
        echo "  GRAND TOTAL: $GRAND_TOTAL baseline reports" >> "$LOG"
        echo "=== Monitor ended at $(date) ===" >> "$LOG"
        exit 0
    fi

    sleep $INTERVAL
done
