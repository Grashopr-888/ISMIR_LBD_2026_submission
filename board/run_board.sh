#!/bin/bash
# Provenance: Grashopr-888/elk-mutek@3baef53:work/elk-granular/board/run_board.sh (changed: glue optional, a site-specific backup step removed)
# ElkGranular on the Stomp — the whole stack, in the devkit's own order: sensei -> sushi -> glue.
# Layout: /udata/elkgranular/{plugins,packs,config,logs[,glue]}   (deploy: board/crossbuild/deploy_serial.sh)
#   ./run_board.sh            # Elk's demo loop -> instrument (first-sound config)
#   ./run_board.sh live       # guitar jack in -> instrument
#   NOGLUE=1 ./run_board.sh   # sushi only (no pots); also the default when no glue/ is installed
# The controls glue (pots and switches over SENSEI/SUSHI gRPC) is not part of this repository; the paper's
# measurements (Experiment 2) do not depend on it. Parameters then keep the config's initial_state.
# Stop with Ctrl-C ONLY. The trap forwards SIGINT to every process; never SIGTERM sushi on this board.
ROOT=/udata/elkgranular
CFG=$ROOT/config/sushi_config_stomp.json
if [ "$1" = "live" ]; then CFG=$ROOT/config/sushi_config_stomp_live.json; shift; fi   # remaining args go to sushi
STAMP=$(date +%Y%m%d-%H%M%S)
# default world: the first pack present; override with ELKGRANULAR_PACK=...
if [ -z "$ELKGRANULAR_PACK" ]; then for cand in field gonza demo; do [ -f $ROOT/packs/$cand.elkgrain ] && { ELKGRANULAR_PACK=$ROOT/packs/$cand.elkgrain; break; }; done; fi
export ELKGRANULAR_PACK
export ELKGRANULAR_STUDENT=$ROOT/packs/student_mlp_mel80.json
export ELKGRANULAR_RIDGE=$ROOT/packs/ridge_pooled.json
# logs go to RAM (/tmp) during the run and are copied to $ROOT/logs on exit: writing to the eMMC while a
# world loads from the same eMMC stalled the hop worker (p99 340 ms, 2026-09-03)
RUNLOGS=/tmp/elkgranular-run-$STAMP; mkdir -p $RUNLOGS
export ELKGRANULAR_LOG=$RUNLOGS/worker-$STAMP.log
export LV2_PATH=/home/mind/plugins/gxplugins-stomp/
mkdir -p $ROOT/logs
echo "pack=$ELKGRANULAR_PACK  config=$CFG  logs=$RUNLOGS -> $ROOT/logs on exit"

# E2c (EVLSTATS=1): a second EVL snapshot at the END of the run, taken before the SIGINT goes out so sushi's
# threads still exist -> evl-$STAMP-end.log (compare ISW with the 25 s one, evl-$STAMP-t025.log).
# E2a (--timing-statistics): sushi writes timings.txt into its cwd at engine teardown, i.e. during `wait`;
# it is renamed timings-$STAMP.txt so the cp lands it in $ROOT/logs with the other logs of this run.
cleanup() { trap - INT; echo "SIGINT -> all"; [ -n "$EVLSTATS" ] && { evl ps -s > $RUNLOGS/evl-$STAMP-end.log 2>&1; evl ps -t >> $RUNLOGS/evl-$STAMP-end.log 2>&1; }; kill -INT 0 2>/dev/null; wait; [ -f $RUNLOGS/timings.txt ] && mv $RUNLOGS/timings.txt $RUNLOGS/timings-$STAMP.txt; cp $RUNLOGS/* $ROOT/logs/ 2>/dev/null; echo "logs copied to $ROOT/logs"; }
trap cleanup INT

if ! pgrep -x sensei >/dev/null; then
    echo "starting sensei"; sensei -f /home/mind/sensei_config.json > $RUNLOGS/sensei-$STAMP.log 2>&1 &
    sleep 2
fi
echo "starting sushi"
# E2a: run sushi from $RUNLOGS so a --timing-statistics run writes timings.txt there (RAM) instead of into the
# home directory. Every path sushi needs is absolute (config, plugin path, wav).
cd $RUNLOGS
sushi -r -c "$CFG" --base-plugin-path="$ROOT/plugins/" -l info --log-file=$RUNLOGS/sushi-$STAMP.log "$@" &   # "$@": extra args (e.g. --timing-statistics) really reach sushi (16 Sep fix)
sleep 3
if [ -z "$NOGLUE" ] && [ -f $ROOT/glue/guru_mapping.py ]; then
    echo "starting glue (guru_mapping.py)"
    PYTHONPATH=/home/mind/guru/src:/home/mind:$ROOT/glue python3 $ROOT/glue/guru_mapping.py > $RUNLOGS/glue-$STAMP.log 2>&1 &
fi
# WATCH=1: mirror bar / world / hold events to the console so a follower on the laptop can watch live
if [ -n "$WATCH" ]; then ( sleep 6; tail -n 0 -F $ELKGRANULAR_LOG 2>/dev/null | grep --line-buffered -E "^bar|^world" ) & ( sleep 8; tail -n 0 -F $RUNLOGS/glue-$STAMP.log 2>/dev/null | grep --line-buffered -E "world ->|hold" ) & fi
# EVLSTATS=1: snapshot EVL thread stats (mode switches = ISW column) 25 s into the run (and again at Ctrl-C, see cleanup)
if [ -n "$EVLSTATS" ]; then ( sleep 25; evl ps -s > $RUNLOGS/evl-$STAMP-t025.log 2>&1; evl ps -t >> $RUNLOGS/evl-$STAMP-t025.log 2>&1 ) & fi
echo "running — Ctrl-C to stop everything; tail -f $ELKGRANULAR_LOG for bars/picks"
wait
