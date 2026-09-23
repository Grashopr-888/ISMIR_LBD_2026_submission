#!/usr/bin/env python3
# Provenance: Grashopr-888/elk-mutek@3baef53:scripts/serial_follow.py (changed: login read from ELK_BOARD_USER / ELK_BOARD_PASSWORD, paths in comments)
"""serial_follow.py — run a board session and STREAM its telemetry live (one line per event, flushed).
    uv run --with pyserial scripts/serial_follow.py --seconds 900 --cmd "WATCH=1 /udata/elkgranular/run_board.sh"
Prints world switches, hold events, errors, and one bar line every ~20 s; SIGINTs the run at the end
and prints the worker SUMMARY. Meant to be the command of a Monitor."""
import argparse, glob, os, re, sys, time
import serial
ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=900); ap.add_argument("--cmd", default="WATCH=1 /udata/elkgranular/run_board.sh")
ap.add_argument("--device"); ap.add_argument("--user", default=os.environ.get("ELK_BOARD_USER", "mind")); ap.add_argument("--password", default=os.environ.get("ELK_BOARD_PASSWORD"), help="board login password; default $ELK_BOARD_PASSWORD (see board/README.md)")
a = ap.parse_args()
dev = a.device or sorted(glob.glob("/dev/cu.usbserial-*"))[0]
port = serial.Serial(dev, 115200, timeout=0.3)
def read_until(markers, timeout):
    buf, t0 = "", time.time()
    while time.time() - t0 < timeout:
        c = port.read(4096).decode("utf-8", "replace")
        if c:
            buf += c
            if any(buf.rstrip().endswith(m.rstrip()) for m in markers): return buf, True
    return buf, False
port.reset_input_buffer(); port.write(b"\r")
out, _ = read_until(("login:", "$ ", "# "), 10)
if out.rstrip().endswith("login:"):
    if not a.password: sys.exit("login prompt: set ELK_BOARD_PASSWORD (or --password) to the dev kit's login")
    port.write((a.user + "\r").encode()); read_until(("assword:",), 10); port.write((a.password + "\r").encode())
    if not read_until(("$ ", "# "), 25)[1]: sys.exit("login failed")
port.write(b"export PS1='ELKFOLLOW$ '; stty columns 200\r"); read_until(("ELKFOLLOW$",), 10)
print(f"START {time.strftime('%H:%M:%S')} {a.cmd}", flush=True)
port.write((a.cmd + "\r").encode())
t0, buf, last_bar = time.time(), "", 0.0
while time.time() - t0 < a.seconds:
    c = port.read(4096).decode("utf-8", "replace")
    if not c: continue
    buf += c
    while "\n" in buf:
        line, buf = buf.split("\n", 1); line = line.strip("\r ")
        if not line: continue
        if re.search(r"ElkGranular: world|^world \d|world -> |hold (ENGAGED|released)|Traceback|\[error\]|SIGINT|pack=", line):
            print(f"{time.strftime('%H:%M:%S')} {line[:150]}", flush=True)
        elif line.startswith("bar ") and time.time() - last_bar > 20:
            print(f"{time.strftime('%H:%M:%S')} {re.sub(r' picks=.*', '', line)[:140]}", flush=True); last_bar = time.time()
port.write(b"\x03"); out, ok = read_until(("ELKFOLLOW$",), 45)
print(f"STOP {time.strftime('%H:%M:%S')} (prompt {'back' if ok else 'missing'})", flush=True)
port.write(b"W=$(ls -t /udata/elkgranular/logs/worker-*.log | head -n 1); grep SUMMARY $W /dev/null | cut -c1-330\r")
out, _ = read_until(("ELKFOLLOW$",), 30)
for l in out.splitlines():
    if "SUMMARY mode" in l and not l.strip().startswith("W="): print(l.strip()[:330], flush=True)
port.close()
