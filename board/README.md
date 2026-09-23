# board/ — building for, deploying to and running on the Elk Stomp Dev Kit

The Elk Stomp Dev Kit (STM32MP157: two Cortex-A7 cores at 800 MHz, no neural processing unit) runs Elk Audio OS with
the SUSHI host. The plugin (`plugin/`) runs inside SUSHI as a VST3 with a 64-sample callback at 48 kHz. Its analysis
runs on a worker thread through Elk's asynchronous-work extension.

| Path | What |
|---|---|
| `crossbuild/` | Docker image (Ubuntu 24.04 arm64, GCC 13 for armhf, JUCE 8.0.12 installed as in Elk's SDK), toolchain file, `build.sh` (Release build of the plugin and `elkgrain_bench` into `plugin/build-stomp/`), `deploy_serial.sh` (copy over the USB serial console). See `crossbuild/README.md` |
| `run_board.sh` | on the board: starts SENSEI and SUSHI with the plugin, logs to RAM, copies them to `/udata/elkgranular/logs` on Ctrl-C. `EVLSTATS=1` adds EVL thread statistics, extra arguments go to SUSHI (`--timing-statistics` was used for Experiment 2) |
| `sushi_config_stomp.json` | Elk's demo loop into the plugin (the Experiment 2 configuration) |
| `sushi_config_stomp_live.json` | the input jack into the plugin |
| `serial/` | USB serial console helpers: `board_probe.py` (is the board up, read-only), `serial_console.py` (run commands, save a transcript), `serial_send.py` (send a file), `serial_follow.py` (follow a run) |

## Steps

```bash
docker build --platform linux/arm64 -t elk-stomp-crossbuild board/crossbuild   # once
board/crossbuild/build.sh
export ELK_BOARD_USER=mind ELK_BOARD_PASSWORD=...                               # the dev kit's login (Elk's documentation)
board/crossbuild/deploy_serial.sh work/demo/demo.elkgrain                       # plugin, configs, weights, packs
uv run --with pyserial board/serial/serial_console.py --cmds-file results/board/cmds/R1-field-f32.txt --out board/logs/run.txt
```

`results/board/cmds/` are the exact command files of Experiment 2 (one run per file: stop everything, run the stack
for 210–330 s with `--timing-statistics` and `EVLSTATS=1`, then print the worker log's SUMMARY and STAGES lines, SUSHI's
processor lines, the timings file and the EVL snapshots). `ELKGRANULAR_TABLE=int8` selects the 8-bit latent table,
`ELKGRANULAR_PACK` the pack.

The serial helpers read the login from `ELK_BOARD_USER` (default `mind`, Elk's documented user) and
`ELK_BOARD_PASSWORD` (no default: use the dev kit's documented password or your own). Nothing here stores a password.

## Not included

- The controls glue that maps the dev kit's pots and switches to the plugin's parameters over SENSEI/SUSHI gRPC (it
  was adapted from an Elk example and uses stubs generated from Elk's protocol files). Experiment 2 does not depend on
  it: without it the parameters keep `initial_state` from the SUSHI config (τ = 0.05, wet 1).
- Packs: build them with `corpus_builder` from your own recordings (notebook 01) and pass them to `deploy_serial.sh`.

## Status of these scripts

The board build of the paper (16 Sep 2026) used these scripts with RTNeural and Elk's header from local checkouts. For
this repository CMake now fetches both at the same commits, and the Docker image already contains `git`. The desktop
build from this tree was verified on 23 Sep 2026. The cross build and deployment were not re-run from this tree.
