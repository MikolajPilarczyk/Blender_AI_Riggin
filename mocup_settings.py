"""
Okno ustawień dla skryptu nagrywającego + wspólny plik ustawień dla Blendera.

- W skrypcie z kamerą naciśnij S, żeby otworzyć okno.
- Można też uruchomić samo okno bez kamery:  python mocap_settings.py
- Każda zmiana zapisuje się automatycznie do  mocap_settings.json  (obok tego pliku).
- Skrypt w Blenderze (blender_mocap_to_rig.py) czyta ten plik i nadpisuje swoje ustawienia.
- Lista rozwijana pokazuje pliki .json z folderu  recordings/  (odświeża się sama).
"""

import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
SETTINGS_FILE = os.path.join(BASE_DIR, "mocap_settings.json")
LATEST = "(najnowsze)"

# (klucz, opis, typ, domyślna [, min, max, krok])
# Klucze MUSZĄ się nazywać tak samo jak stałe w blender_mocap_to_rig.py
TABS = [
    ("Ogólne", [
        ("MIRROR", "Tryb lustra (podnosisz prawą rękę -> postać podnosi lewą)", "bool", True),
        ("USE_HANDS", "Dłonie i palce", "bool", True),
        ("HANDEDNESS_HIGH_IS_RIGHT", "Strony dłoni: wartość > 0.5 = prawa (tylko stare nagrania)", "bool", True),
        ("SMOOTHING", "Wygładzanie (1 = brak, mniej = gładziej, ale wolniej)", "float", 0.05, 0.01, 1.0, 0.01),
        ("MIN_PRESENCE", "Minimalna pewność punktu", "float", 0.5, 0.0, 1.0, 0.05),

    ]),
    ("Tułów i głowa", [
        ("UPRIGHT_BLEND", "Prostowanie tułowia (0 = surowe, 1 = pion)", "float", 0.5, 0.0, 1.0, 0.05),
        ("MAX_LEAN_DEG", "Maksymalny przechył tułowia (stopnie)", "float", 12, 0, 60, 1),
        ("TORSO_DEPTH", "Tłumienie głębi tułowia (1 = bez tłumienia)", "float", 0.4, 0.0, 1.0, 0.05),
        ("HIP_ROLL_DAMP", "Tłumienie przechyłu bioder (1 = bez tłumienia)", "float", 0.3, 0.0, 1.0, 0.05),
        ("HEAD_UPRIGHT_BLEND", "Prostowanie głowy", "float", 0.3, 0.0, 1.0, 0.05),
        ("MAX_HEAD_TILT_DEG", "Maksymalne pochylenie głowy (stopnie)", "float", 25, 0, 90, 1),
        ("HEAD_DEPTH", "Tłumienie głębi głowy", "float", 0.5, 0.0, 1.0, 0.05),
    ]),
    ("Nogi i kamera", [
        ("LEG_CORRECTION", "Siła ściągania nóg do naturalnego rozstawu", "float", 0.6, 0.0, 1.0, 0.05),
        ("LEG_NATURAL_SPLAY_DEG", "Naturalny rozstaw nóg (stopnie na zewnątrz)", "float", 4, 0, 20, 0.5),
        ("LEG_MAX_INWARD_DEG", "Maks. zejście nogi do środka (stopnie)", "float", 3, 0, 20, 0.5),
        ("LEG_MAX_OUTWARD_DEG", "Maks. odejście nogi na zewnątrz (stopnie)", "float", 25, 0, 60, 1),
        ("LEG_DEPTH_SCALE", "Tłumienie głębi nóg", "float", 0.5, 0.0, 1.0, 0.05),
        ("FOOT_SPLAY_SCALE", "Tłumienie rozchylania stóp", "float", 0.4, 0.0, 1.0, 0.05),
        ("CAMERA_PITCH_DEG", "Korekta pochylenia kamery (stopnie, +/-)", "float", 0, -30, 30, 1),
    ]),
]

DEFAULTS = {item[0]: item[3] for _, items in TABS for item in items}


# ---------------------------------------------------------------- pliki
def list_recordings():
    """Nazwy plików .json z folderu recordings, od najnowszego."""
    if not os.path.isdir(RECORDINGS_DIR):
        return []
    files = [n for n in os.listdir(RECORDINGS_DIR) if n.lower().endswith(".json")]
    files.sort(key=lambda n: os.path.getmtime(os.path.join(RECORDINGS_DIR, n)), reverse=True)
    return files


def load_settings():
    data = dict(DEFAULTS)
    data["recording"] = LATEST
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if k in data:
                    data[k] = v
        except (OSError, ValueError) as e:
            print(f"[ustawienia] Nie można wczytać {SETTINGS_FILE}: {e}")
    return data


def save_settings(data):
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, SETTINGS_FILE)   # zapis atomowy: Blender nigdy nie zobaczy połowy pliku


# ---------------------------------------------------------------- okno
class SettingsWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ustawienia mocap -> Blender")
        self.root.geometry("500x720")
        self._ready = False
        self._after_id = None
        self.values = load_settings()
        self.vars = {}

        # --- wybór nagrania
        top = ttk.LabelFrame(self.root, text="Nagranie do importu w Blenderze", padding=8)
        top.pack(fill="x", padx=8, pady=(8, 4))
        row = ttk.Frame(top)
        row.pack(fill="x")
        self.rec_var = tk.StringVar(value=self.values.get("recording") or LATEST)
        self.combo = ttk.Combobox(row, textvariable=self.rec_var, state="readonly", values=[LATEST])
        self.combo.pack(side="left", fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self._changed())
        ttk.Button(row, text="Odśwież", command=self._refresh_once).pack(side="left", padx=(6, 0))
        if hasattr(os, "startfile"):
            ttk.Button(row, text="Folder", command=self._open_folder).pack(side="left", padx=(6, 0))
        self.info = ttk.Label(top, text="")
        self.info.pack(anchor="w", pady=(6, 0))

        # --- zakładki z ustawieniami
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=8, pady=4)
        for title, items in TABS:
            frame = ttk.Frame(nb, padding=10)
            nb.add(frame, text=title)
            self._build(frame, items)

        # --- dół okna
        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(4, 0))
        ttk.Button(bottom, text="Przywróć domyślne", command=self._reset).pack(side="left")
        self.status = ttk.Label(bottom, text="", foreground="#2a7a2a")
        self.status.pack(side="right")
        ttk.Label(self.root, foreground="#666666",
                  text="Po zmianie uruchom skrypt w Blenderze ponownie (Alt+P).").pack(pady=(4, 8))

        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self._refresh()
        self._ready = True
        self._save()   # utwórz plik od razu, żeby Blender go widział

    # ---- budowanie widżetów
    def _build(self, parent, items):
        for item in items:
            key, label, kind, default = item[:4]
            value = self.values.get(key, default)
            if kind == "bool":
                var = tk.BooleanVar(value=bool(value))
                ttk.Checkbutton(parent, text=label, variable=var,
                                command=self._changed).pack(anchor="w", pady=3)
            elif kind == "str":
                var = tk.StringVar(value=str(value))
                ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
                ttk.Entry(parent, textvariable=var).pack(fill="x")
                var.trace_add("write", lambda *_: self._changed())
            else:
                lo, hi, step = item[4:7]
                var = tk.DoubleVar(value=float(value))
                ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
                tk.Scale(parent, from_=lo, to=hi, resolution=step, orient="horizontal",
                         variable=var, command=lambda _v: self._changed()).pack(fill="x")
            self.vars[key] = var

    # ---- zapisywanie (z opóźnieniem, żeby nie pisać na dysk przy każdym pikselu suwaka)
    def _changed(self):
        if not self._ready:
            return
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self._after_id = self.root.after(250, self._save)

    def _save(self):
        self._after_id = None
        data = {}
        for key, var in self.vars.items():
            v = var.get()
            data[key] = round(v, 4) if isinstance(v, float) else v
        data["recording"] = self.rec_var.get() or LATEST
        try:
            save_settings(data)
            self.status.config(text=f"Zapisano {time.strftime('%H:%M:%S')}", foreground="#2a7a2a")
        except OSError as e:
            self.status.config(text=f"Błąd zapisu: {e}", foreground="#b00020")

    def _reset(self):
        self._ready = False
        for key, var in self.vars.items():
            var.set(DEFAULTS[key])
        self._ready = True
        self._save()

    # ---- lista nagrań
    def _refresh_once(self):
        self._refresh(repeat=False)

    def _refresh(self, repeat=True):
        files = list_recordings()
        values = [LATEST] + files
        if tuple(self.combo["values"]) != tuple(values):
            self.combo["values"] = values
        if files:
            self.info.config(text=f"Nagrań: {len(files)}  |  najnowsze: {files[0]}")
        else:
            self.info.config(text=f"Brak nagrań w {RECORDINGS_DIR}")
        if repeat:
            self.root.after(2000, self._refresh)

    def _open_folder(self):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.startfile(RECORDINGS_DIR)


# ---------------------------------------------------------------- uruchamianie
_thread = None


def _run():
    try:
        SettingsWindow().root.mainloop()
    except Exception as e:   # okno ustawień nie może przewrócić nagrywania
        print(f"[ustawienia] Błąd okna ustawień: {e}")


def open_settings_window():
    """Otwiera okno w osobnym wątku (nie spowalnia pętli z kamerą). Zwraca False, jeśli już otwarte."""
    global _thread
    if _thread is not None and _thread.is_alive():
        return False
    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()
    return True


if __name__ == "__main__":
    SettingsWindow().root.mainloop()