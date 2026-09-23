#!/usr/bin/env python3
# Provenance: Grashopr-888/elk-mutek@3baef53:scripts/serial_console.py (changed: login read from ELK_BOARD_USER / ELK_BOARD_PASSWORD, paths in comments)
"""Scripted serial console for the Elk Stomp dev board (T01 bring-up capture).

Logs in over the USB TO UART console (115200, mind/elk) and runs a fixed,
READ-ONLY command list, writing a full transcript to stdout/--out.

Usage:
  uv run --with pyserial scripts/serial_console.py [--device /dev/cu.usbserial-XXXX]
      [--out board/logs/board-serial-transcript.txt] [--cmd "single command"]

Device default: the lowest-numbered /dev/cu.usbserial-* (the console is the
lowest of the four UARTs the board exposes). Exit picocom-style manual
sessions with Ctrl-A Ctrl-X; this script needs no interaction.
"""
import argparse, glob, os, sys, time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial missing — run via: uv run --with pyserial scripts/serial_console.py")

# READ-ONLY capture set (T01). No sudo, no writes, no service restarts.
T01_COMMANDS = [
    "cat /etc/os-release",
    "uname -a",
    "cat /etc/hostname; date",
    "sushi --version 2>&1 | head -5",
    "cat /udata/.elk-system/elk-system.conf 2>/dev/null",
    "head -30 /proc/cpuinfo",
    "free -h; df -h | head -10",
    "ip addr",
    "ps aux | head -25",
    "ls -la /home/mind/",
    "ls -la /home/mind/bin/ 2>/dev/null",
    "ls -laR /home/mind/plugins/ 2>/dev/null | head -200",
    "find /home/mind -maxdepth 3 \\( -name '*.json' -o -name '*.vst3' -o -name '*.lv2' \\) 2>/dev/null | head -60",
    "find /usr/lib /usr/share -maxdepth 4 \\( -name '*.vst3' -o -name '*.lv2' \\) 2>/dev/null | head -60",
    "for f in /home/mind/*.json /home/mind/config*/*.json; do echo \"== $f\"; head -80 \"$f\" 2>/dev/null; done",
    "systemctl status sushi --no-pager 2>/dev/null | head -12",
    "journalctl -u sushi --no-pager 2>/dev/null | tail -20",
    "evl ps -t 2>&1 | head -15",
    "ls /tmp/",
]

PROMPT_TAILS = ("$ ", "# ", "$", "#")
PROMPTS = ("ELKPROMPT$", "elk-stomp-dev:~$", "elk-stomp-dev:~#")   # our marker, or the board's default prompt in $HOME


def read_until(port, markers, timeout, quiet=0.6):
    """Read until any marker appears at buffer end, or quiet-period after data, or timeout."""
    buf, last_data, start = "", time.time(), time.time()
    while time.time() - start < timeout:
        chunk = port.read(4096).decode("utf-8", "replace")
        if chunk:
            buf += chunk
            last_data = time.time()
            tail = buf.rstrip()[-40:]
            if any(tail.endswith(m) for m in markers):
                return buf, True
        elif buf and time.time() - last_data > quiet:
            return buf, False
        else:
            time.sleep(0.05)
    return buf, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--user", default=os.environ.get("ELK_BOARD_USER", "mind"))
    ap.add_argument("--password", default=os.environ.get("ELK_BOARD_PASSWORD"), help="board login password; default $ELK_BOARD_PASSWORD (see board/README.md)")
    ap.add_argument("--out")
    ap.add_argument("--cmd", help="run just this one command instead of the T01 set")
    ap.add_argument("--cmds-file", help="file with one command per line (BusyBox syntax!)")
    a = ap.parse_args()

    dev = a.device or (sorted(glob.glob("/dev/cu.usbserial-*")) or [None])[0]
    if not dev:
        sys.exit("no /dev/cu.usbserial-* device found — is the Mac on the USB TO UART port?")
    log_lines = [f"# Elk Stomp serial transcript — {time.strftime('%Y-%m-%d %H:%M:%S')} — {dev}\n"]

    def log(s):
        log_lines.append(s)
        print(s, end="" if s.endswith("\n") else "\n")

    port = serial.Serial(dev, a.baud, timeout=0.2)
    port.reset_input_buffer()

    # Wake the console; find out if we're at login: or already at a shell.
    port.write(b"\r")
    out, _ = read_until(port, ("login: ", "login:") + PROMPT_TAILS, timeout=8)
    if "login:" in out.split("\n")[-1] or out.rstrip().endswith("login:"):
        port.write((a.user + "\r").encode())
        out2, _ = read_until(port, ("Password:", "assword:"), timeout=8)
        if not a.password:
            sys.exit("login prompt: set ELK_BOARD_PASSWORD (or --password) to the dev kit's login")
        port.write((a.password + "\r").encode())
        out3, ok = read_until(port, PROMPT_TAILS, timeout=10)
        log(out + out2 + out3)
        if not ok and "incorrect" in out3.lower():
            sys.exit("login failed")
    else:
        log(out)  # already at a shell (or console echoed something else)

    def paced_write(s, delay=0.004):
        """BusyBox/UART-safe: write char-by-char so nothing gets dropped on the wire."""
        for ch in s.encode():
            port.write(bytes([ch]))
            time.sleep(delay)

    # Make the prompt detectable and the terminal wide. Paced and retried: on 16 Sep the unpaced write right
    # after login was lost (the shell was still printing), every later command then waited 25 s for a prompt
    # that never came and was SIGINT'd. PROMPTS also accepts the board's default prompt as a fallback.
    for _ in range(3):
        paced_write("export PS1='ELKPROMPT$ '; stty columns 200\r")
        _, ok = read_until(port, ("ELKPROMPT$",), timeout=5)
        if ok:
            break
        port.write(b"\r"); time.sleep(0.5); port.reset_input_buffer()
    else:
        log("[serial_console] could not set ELKPROMPT$; falling back to the default prompt marker")

    def read_for(secs):
        """Collect everything for a fixed duration (long-running foreground process)."""
        buf, start = "", time.time()
        while time.time() - start < secs:
            chunk = port.read(4096).decode("utf-8", "replace")
            if chunk:
                buf += chunk
            else:
                time.sleep(0.05)
        return buf

    if a.cmds_file:
        with open(a.cmds_file) as f:
            cmds = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    else:
        cmds = [a.cmd] if a.cmd else T01_COMMANDS
    for c in cmds:
        # "RUN:<seconds>:<command>" — start it, collect for N s, then Ctrl-C (SIGINT) it.
        if c.startswith("RUN:"):
            _, secs, cmd = c.split(":", 2)
            log(f"\n\n===== RUN {secs}s: {cmd}\n")
            paced_write(cmd + "\r")
            log(read_for(float(secs)))
            port.write(b"\x03")  # ^C → SIGINT to foreground job (the sanctioned stop)
            out, _ = read_until(port, PROMPTS, timeout=30)
            log("\n--- after SIGINT ---\n" + out.replace("ELKPROMPT$", "").replace("mind@elk-stomp-dev:~$", "").strip())
            continue
        log(f"\n\n===== CMD: {c}\n")
        paced_write(c + "\r")
        out, ok = read_until(port, PROMPTS, timeout=25)
        if not ok:
            # No prompt back: the command is hung or reading the console (a grep/cat with no
            # file argument does exactly that and then eats every later command). SIGINT it.
            port.write(b"\x03")
            out2, ok2 = read_until(port, PROMPTS, timeout=10)
            out += out2
            log("\n[serial_console] no prompt after 25 s -> sent SIGINT; prompt " + ("recovered" if ok2 else "STILL missing"))
        # strip echoed command + trailing prompt for readability, keep raw otherwise
        log(out.replace("ELKPROMPT$", "").replace("mind@elk-stomp-dev:~$", "").strip())

    port.close()
    if a.out:
        with open(a.out, "w") as f:
            f.write("".join(l if l.endswith("\n") else l + "\n" for l in log_lines))
        print(f"\n[transcript saved → {a.out}]")


if __name__ == "__main__":
    main()
