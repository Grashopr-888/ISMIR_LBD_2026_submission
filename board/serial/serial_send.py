#!/usr/bin/env python3
# Provenance: Grashopr-888/elk-mutek@3baef53:scripts/serial_send.py (changed: login read from ELK_BOARD_USER / ELK_BOARD_PASSWORD, paths in comments)
"""serial_send.py — copy ONE file to the Elk Stomp over the USB-TO-UART console (no network needed).

    uv run --with pyserial scripts/serial_send.py <local_file> <remote_path> [--device /dev/cu.usbserial-XXXX0]
                                                  [--chunk 4096] [--pace 0.0]

How: logs in (ELK_BOARD_USER / ELK_BOARD_PASSWORD, or --user/--password), turns off console echo,
runs a Python base64 decoder on the board (it has no `base64` binary), streams the file as base64 lines at the
full 115200 baud, ends with Ctrl-D, restores echo, verifies sha256 on both sides, then
renames .part -> final. Exit 0 only on a checksum match. ~8 KB/s of payload.
Writes only under the path you give (use /udata/xfer/... — the rootfs has ~200 MB free).
"""
import argparse, base64, glob, hashlib, os, sys, time

try:
    import serial
except ImportError:
    sys.exit("pyserial missing — run via: uv run --with pyserial scripts/serial_send.py")

PROMPT = "ELKSEND$ "


def read_until(port, markers, timeout, quiet=0.8):
    buf, last, start = "", time.time(), time.time()
    while time.time() - start < timeout:
        chunk = port.read(4096).decode("utf-8", "replace")
        if chunk:
            buf += chunk; last = time.time()
            if any(buf.rstrip().endswith(m.rstrip()) for m in markers):
                return buf, True
        elif buf and time.time() - last > quiet:
            return buf, False
    return buf, False


def shell(port, cmd, timeout=20):
    port.write((cmd + "\r").encode())
    out, ok = read_until(port, (PROMPT,), timeout)
    if not ok:
        raise RuntimeError(f"no prompt after: {cmd}\n{out[-400:]}")
    lines = out.replace("\r", "").split("\n")
    return "\n".join(l for l in lines[1:] if l.strip() and not l.strip().endswith(PROMPT.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("local"); ap.add_argument("remote")
    ap.add_argument("--device"); ap.add_argument("--user", default=os.environ.get("ELK_BOARD_USER", "mind")); ap.add_argument("--password", default=os.environ.get("ELK_BOARD_PASSWORD"), help="board login password; default $ELK_BOARD_PASSWORD (see board/README.md)")
    ap.add_argument("--chunk", type=int, default=4096, help="bytes of base64 per write")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to sleep between chunks")
    a = ap.parse_args()

    data = open(a.local, "rb").read()
    sha = hashlib.sha256(data).hexdigest()
    b64 = base64.encodebytes(data)                      # 76-char lines, '\n' terminated
    ports = sorted(glob.glob("/dev/cu.usbserial-*"))
    dev = a.device or (ports[0] if ports else None)
    if not dev:
        sys.exit("no /dev/cu.usbserial-* device")
    port = serial.Serial(dev, 115200, timeout=0.3)
    port.reset_input_buffer()

    # wake + login
    port.write(b"\r")
    out, _ = read_until(port, ("login:", "$ ", "# "), 10)
    if out.rstrip().endswith("login:"):
        port.write((a.user + "\r").encode()); read_until(port, ("assword:",), 10)
        if not a.password:
            sys.exit("login prompt: set ELK_BOARD_PASSWORD (or --password) to the dev kit's login")
        port.write((a.password + "\r").encode())
        out, ok = read_until(port, ("$ ", "# "), 25)
        if not ok:
            sys.exit("login failed")
    port.write((f"export PS1='{PROMPT}'; stty columns 200\r").encode())
    read_until(port, (PROMPT,), 10)

    remote_dir = os.path.dirname(a.remote) or "."
    shell(port, f"mkdir -p {remote_dir}")
    shell(port, f"rm -f {a.remote}.part")
    # -echo: the console must not echo 100k+ lines back; -ixon: no XON/XOFF surprises
    # the board has no `base64` binary but has Python 3.12: decode with it (newlines are ignored by b64decode)
    port.write((f"stty -echo -ixon; python3 -c 'import sys,base64; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))' "
                f"> {a.remote}.part\r").encode())
    time.sleep(0.5); port.reset_input_buffer()

    t0 = time.time(); sent = 0
    for i in range(0, len(b64), a.chunk):
        port.write(b64[i:i + a.chunk]); port.flush(); sent += min(a.chunk, len(b64) - i)
        if a.pace: time.sleep(a.pace)
        if (i // a.chunk) % 50 == 0:
            print(f"\r  {sent/len(b64)*100:5.1f}%  {sent/1024/(time.time()-t0+1e-9):5.1f} KB/s (base64)", end="", flush=True)
    port.write(b"\x04")                                # EOF at line start -> base64 -d exits
    port.flush()
    out, ok = read_until(port, (PROMPT,), 60)
    print(f"\r  100.0%  {len(b64)/1024/(time.time()-t0):5.1f} KB/s (base64) in {time.time()-t0:.0f} s")
    shell(port, "stty echo ixon")
    if "invalid input" in out.lower() or "error" in out.lower():
        print("board reported:", out.strip()[-300:])

    # sha256 of a big file on the Cortex-A7 takes a while (~1 s per MB): scale the timeout
    remote_sha = shell(port, f"sha256sum {a.remote}.part", timeout=max(30, int(len(data) / 1e6 * 2) + 30)).split()[0]
    size = shell(port, f"stat -c %s {a.remote}.part").strip()
    if remote_sha == sha:
        shell(port, f"mv {a.remote}.part {a.remote}")
        if os.access(a.local, os.X_OK):                   # keep the executable bit (scripts)
            shell(port, f"chmod +x {a.remote}")
        print(f"OK  {a.remote}  {size} bytes  sha256 {sha[:16]}…  ({len(data)/1024:.0f} KB payload)")
        port.close(); sys.exit(0)
    print(f"MISMATCH: local {sha[:16]}… remote {remote_sha[:16]}… (remote size {size}, local {len(data)}); left {a.remote}.part in place")
    port.close(); sys.exit(1)


if __name__ == "__main__":
    main()
