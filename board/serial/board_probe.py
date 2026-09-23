#!/usr/bin/env python3
# Provenance: Grashopr-888/elk-mutek@3baef53:scripts/board_probe.py (verbatim)
"""board_probe.py — is the Elk Stomp on?  Read-only console probe, NO login.

Run this every time before working with the board:

    uv run --with pyserial scripts/board_probe.py            # auto-picks the console (lowest usbserial)
    uv run --with pyserial scripts/board_probe.py --seconds 10 --device /dev/cu.usbserial-21400

Exit codes:  0 = ON (login prompt or shell seen)   2 = booting/active, no prompt yet
             3 = silent (off, unpowered, or wrong port)   4 = no usbserial device / port busy
It only sends one carriage return (re-draws the prompt); it never types a username or password.
"""
import argparse, glob, re, sys, time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing — run via: uv run --with pyserial scripts/board_probe.py")

ap = argparse.ArgumentParser()
ap.add_argument("--device", help="console port; default = lowest-numbered /dev/cu.usbserial-*")
ap.add_argument("--seconds", type=float, default=6.0, help="how long to listen")
a = ap.parse_args()

ports = sorted(glob.glob("/dev/cu.usbserial-*"))
dev = a.device or (ports[0] if ports else None)
if not dev:
    print("NO CONSOLE — no /dev/cu.usbserial-* at all: USB cable not in the 'USB TO UART' port, or not enumerated")
    sys.exit(4)
try:
    s = serial.Serial(dev, 115200, timeout=0.5)
except serial.SerialException as e:
    print(f"PORT BUSY/UNAVAILABLE {dev}: {e}  (picocom still open? exit it with C-a C-x)")
    sys.exit(4)

with s:
    s.reset_input_buffer()
    time.sleep(0.2)
    s.write(b"\r")                      # nudges a login prompt / shell to redraw; harmless
    buf, t0 = b"", time.time()
    while time.time() - t0 < a.seconds:
        chunk = s.read(4096)
        if chunk:
            buf += chunk
            if b"login:" in buf:
                break

txt = buf.decode("utf-8", "replace")
lines = [l for l in txt.strip().splitlines() if l.strip()]
tail = "\n".join(lines[-6:])
if "login:" in txt:
    print(f"ON — login prompt on {dev} (log in yourself: picocom -b 115200 {dev}, user mind)\n---\n{tail}")
    sys.exit(0)
if re.search(r"[$#] ?$", txt.rstrip()):
    print(f"ON — a shell is already logged in on {dev} (a previous session left it open)\n---\n{tail}")
    sys.exit(0)
if lines:
    print(f"BOOTING/ACTIVE — output but no prompt yet on {dev}; boot takes ~30 s, re-run\n---\n{tail}")
    sys.exit(2)
print(f"SILENT — no bytes in {a.seconds:.0f} s on {dev}: board off / no 9 V power / wrong port (console = lowest-numbered of {', '.join(ports)})")
sys.exit(3)
