#!/bin/bash -e
# Provenance: Grashopr-888/elk-mutek@3baef53:tools/crossbuild/deploy_serial.sh (changed: repository paths; the controls glue and gRPC stubs are not shipped, packs and weights come from arguments)
# Deploy the cross-built ElkGranular.vst3 + board configs + run script to the Elk Stomp over the USB-TO-UART
# console (no network). ~8 KB/s: the stripped plugin takes a few minutes.
#   board/crossbuild/deploy_serial.sh [pack.elkgrain ...]
# The student and ridge weights (weights/*.json) always go along; packs you built yourself are optional
# (large packs are faster to copy with a USB stick or network dongle).
# Needs ELK_BOARD_USER / ELK_BOARD_PASSWORD (the dev kit's login, see board/README.md) and pyserial.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PLUGIN="$ROOT/plugin"
BUNDLE=$(find "$PLUGIN/build-stomp/ElkGranular_artefacts" -type d -name "ElkGranular.vst3" | head -1)
[ -d "$BUNDLE" ] || { echo "no ElkGranular.vst3 under plugin/build-stomp — run board/crossbuild/build.sh first"; exit 1; }
STAGE=$(mktemp -d); mkdir -p "$STAGE/elkgranular/plugins" "$STAGE/elkgranular/config" "$STAGE/elkgranular/packs"
cp -R "$BUNDLE" "$STAGE/elkgranular/plugins/"
cp "$ROOT/board/sushi_config_stomp.json" "$ROOT/board/sushi_config_stomp_live.json" "$STAGE/elkgranular/config/"
cp "$ROOT/board/run_board.sh" "$STAGE/elkgranular/"
cp "$ROOT/weights/student_mlp_mel80.json" "$ROOT/weights/ridge_pooled.json" "$STAGE/elkgranular/packs/"
for PK in "$@"; do cp "$PK" "$STAGE/elkgranular/packs/"; done
if [ -f "$PLUGIN/build-stomp/elkgrain_bench" ]; then mkdir -p "$STAGE/elkgranular/bin" && cp "$PLUGIN/build-stomp/elkgrain_bench" "$STAGE/elkgranular/bin/"; fi   # microbench -> /udata/elkgranular/bin/
COPYFILE_DISABLE=1 tar --no-xattrs -C "$STAGE" -czf "$STAGE/elkgranular-deploy.tgz" elkgranular   # no macOS ._ AppleDouble files
ls -la "$STAGE/elkgranular-deploy.tgz"
mkdir -p "$ROOT/board/logs"
uv run --with pyserial "$ROOT/board/serial/serial_send.py" "$STAGE/elkgranular-deploy.tgz" /udata/xfer/elkgranular-deploy.tgz
uv run --with pyserial "$ROOT/board/serial/serial_console.py" --cmd "cd /udata && tar -xzf xfer/elkgranular-deploy.tgz && chmod +x elkgranular/run_board.sh && ls -la elkgranular elkgranular/plugins/ElkGranular.vst3/Contents/*/ && file elkgranular/plugins/ElkGranular.vst3/Contents/*/ElkGranular.so 2>/dev/null; ls elkgranular/packs" --out "$ROOT/board/logs/deploy-$(date +%Y%m%d-%H%M).txt" | tail -15
rm -rf "$STAGE"
