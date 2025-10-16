# controlcentre.py (updated strict wakeword + blocking mode)
"""
Voice-first controller (updated):
 - Wake word strictly requires "jarvis" at start or alone.
 - Avoids false triggers from normal commands being misheard.
 - Controller pauses listening while a feature is active, resumes after it exits.

Usage: python controlcentre.py
"""

import os
import sys
import subprocess
import time
import traceback
import importlib
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PYTHON_EXE = sys.executable
WAKEWORD = os.getenv("WAKEWORD", "jarvis").lower()

MODULE_NAMES = {
    "face": "facerecognition.py",
    "hand": "objectdetection.py",
    "object": "objectdetection.py",
    "servo": "objectdetection.py",
}

running_procs = {}

try:
    dynamicbot = importlib.import_module("dynamicbot")
except Exception:
    print("[controller] Failed to import dynamicbot. Traceback:")
    traceback.print_exc()
    dynamicbot = None


def safe_speak(text: str) -> None:
    if dynamicbot is not None and hasattr(dynamicbot, "speak"):
        try:
            dynamicbot.speak(text)
            return
        except Exception as e:
            print("[controller] dynamicbot.speak failed:", e)
    print("BOT:", text)


def start_subprocess_module(key: str) -> Optional[subprocess.Popen]:
    key_lower = key.lower()
    script = MODULE_NAMES.get(key_lower)
    if not script:
        return None
    script_path = os.path.join(BASE_DIR, script)
    if not os.path.exists(script_path):
        print(f"[controller] Script not found: {script_path}")
        return None
    proc = running_procs.get(key_lower)
    if proc and proc.poll() is None:
        print(f"[controller] {key_lower} already running (pid {proc.pid})")
        return proc
    try:
        p = subprocess.Popen([PYTHON_EXE, script_path], cwd=BASE_DIR)
        running_procs[key_lower] = p
        print(f"[controller] Started {key_lower} -> PID {p.pid}")
        return p
    except Exception as e:
        print("[controller] Failed to start subprocess:", e)
        return None


def stop_subprocess_module(key: str) -> bool:
    key_lower = key.lower()
    proc = running_procs.get(key_lower)
    if not proc:
        return False
    if proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"[controller] Stopped {key_lower}")
        except Exception as e:
            print("[controller] Failed to stop:", e)
            return False
    running_procs.pop(key_lower, None)
    return True


def parse_start_stop_command(text: str):
    if not text:
        return (None, None)
    t = text.lower()
    action = None
    if any(w in t for w in ["start", "run", "open", "launch"]):
        action = "start"
    if any(w in t for w in ["stop", "close", "terminate", "quit", "shutdown", "end"]):
        action = "stop"
    target = None
    if "chat" in t or "assistant" in t or "bot" in t:
        target = "chatbot"
    elif "face" in t or "recognize" in t or "recognition" in t:
        target = "face"
    elif any(w in t for w in ["hand", "servo", "object", "finger"]):
        target = "hand"
    if "status" in t:
        return ("status", None)
    return (action, target)


def listen_for_wakeword_with_detector(detector) -> bool:
    try:
        detector.listen()
        return True
    except Exception as e:
        print("[controller] detector.listen() raised:", e)
        return False


def listen_for_wakeword_fallback(wakeword: str = WAKEWORD, max_attempts: Optional[int] = None) -> bool:
    if dynamicbot is None or not hasattr(dynamicbot, "listen"):
        print("[controller] No dynamicbot.listen() available for fallback wakeword.")
        return False
    attempts = 0
    while True:
        if max_attempts is not None and attempts >= max_attempts:
            return False
        attempts += 1
        try:
            text = dynamicbot.listen(timeout=2, phrase_time_limit=2)
        except TypeError:
            text = dynamicbot.listen()
        except Exception as e:
            print("[controller] listen() error in wakeword fallback:", e)
            text = None
        if text:
            txt = text.lower().strip()
            if txt == wakeword or txt.startswith(wakeword + " "):
                print(f"[controller] Wake word detected: {txt}")
                return True
            else:
                print(f"[controller] ignored phrase during wake listening: {txt}")
        time.sleep(0.15)


def controller_listen_for_single_command() -> Optional[str]:
    used_detector = None
    if dynamicbot is not None and hasattr(dynamicbot, "WakeWordDetector"):
        try:
            used_detector = dynamicbot.WakeWordDetector()
        except Exception as e:
            print("[controller] dynamicbot.WakeWordDetector init failed:", e)
            used_detector = None
    try:
        print(f"[controller] Waiting for wake word (say '{WAKEWORD}')...")
        if used_detector is not None:
            ok = listen_for_wakeword_with_detector(used_detector)
        else:
            ok = listen_for_wakeword_fallback(WAKEWORD)
        if not ok:
            return None
    except Exception as e:
        print("[controller] Wake listen raised:", e)
        return None
    safe_speak("Yes?")
    if dynamicbot is not None and hasattr(dynamicbot, "listen"):
        try:
            try:
                cmd = dynamicbot.listen(timeout=4, phrase_time_limit=6)
            except TypeError:
                cmd = dynamicbot.listen()
            return cmd
        except Exception as e:
            print("[controller] dynamicbot.listen() error:", e)
            return None
    else:
        print("[controller] dynamicbot.listen() unavailable.")
        return None


def main_voice_controller():
    if dynamicbot is None:
        print("[controller] dynamicbot missing — cannot run voice controller.")
        safe_speak("Controller cannot start voice mode because dynamicbot is missing.")
        return
    safe_speak(f"Controller ready. Say the wake word '{WAKEWORD}' and then say start or stop followed by chatbot, face, or hand.")
    try:
        while True:
            cmd = controller_listen_for_single_command()
            if cmd is None:
                continue
            cmd = cmd.lower().strip()
            print("[controller] Heard command:", cmd)
            if any(x in cmd for x in ["exit controller", "shutdown controller", "quit controller", "stop controller"]):
                safe_speak("Controller shutting down. Goodbye.")
                break
            action, target = parse_start_stop_command(cmd)
            if action == "status":
                parts = []
                for k in set(MODULE_NAMES.keys()):
                    p = running_procs.get(k)
                    state = "running" if p and p.poll() is None else "stopped"
                    parts.append(f"{k} is {state}")
                safe_speak("Status: " + ". ".join(parts))
                continue
            if action is None or target is None:
                safe_speak("I only listen for start or stop commands for chatbot, face, or hand.")
                continue
            if action == "start":
                if target == "chatbot":
                    safe_speak("Starting assistant.")
                    try:
                        if hasattr(dynamicbot, "bot_main"):
                            dynamicbot.bot_main()
                        else:
                            safe_speak("Assistant function not found.")
                    except Exception as e:
                        print("[controller] chatbot raised exception:", e)
                        traceback.print_exc()
                        safe_speak("Assistant crashed. Returning to controller.")
                    safe_speak("Returned to controller. Say the wake word to give another command.")
                    continue
                else:
                    proc = start_subprocess_module(target)
                    if proc:
                        safe_speak(f"Starting {target}. Controller will pause until it stops.")
                        proc.wait()  # block until subprocess exits
                        safe_speak(f"{target} finished. Controller listening again.")
                    else:
                        safe_speak(f"Couldn't start {target}.")
                    continue
            if action == "stop":
                if target == "chatbot":
                    safe_speak("If the assistant is running, say bye to it. If you started a chatbot as a subprocess, say stop chatbot again.")
                    continue
                else:
                    ok = stop_subprocess_module(target)
                    safe_speak(f"Stopped {target}." if ok else f"{target} was not running.")
                    continue
    except KeyboardInterrupt:
        print("Controller interrupted by user.")
    finally:
        for k in list(running_procs.keys()):
            stop_subprocess_module(k)


if __name__ == "__main__":
    main_voice_controller()
