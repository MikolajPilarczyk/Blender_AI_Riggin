"""
Przenosi nagranie mocap (JSON z pose_recorder.py) BEZPOŚREDNIO na kości rigu Mixamo.

Bez Empty, bez constraintów, bez bake'owania: skrypt liczy kierunki kości z punktów
i zapisuje klatki kluczowe rotacji na kościach. Postać zostaje w miejscu (nie przesuwa się),
rusza się tylko szkielet: tułów, głowa, ręce, nogi, dłonie i palce.

Brak wykrycia (pose = None / niska pewność punktu) => kość zostaje w ostatniej pozycji.

Użycie: Blender -> Scripting -> otwórz plik -> ustaw PATH -> Run Script (Alt+P)
"""

import bpy
import json
import math
from mathutils import Vector, Matrix

# ====================== USTAWIENIA ======================
PATH = r"C:\Users\uczen\PycharmProjects\WelcomeScreen\recordings\mocap_20261105_143735.json"
ARMATURE_NAME = "Armature"
PREFIX = "mixamorig:"

# True  = postać działa jak LUSTRO (podnosisz prawą rękę -> postać podnosi swoją lewą)
# False = postać NAŚLADUJE ciebie (podnosisz prawą -> postać podnosi prawą)
MIRROR = True

SMOOTHING = 0.3       # 1.0 = bez wygładzania, mniejsze = gładziej (ale większe opóźnienie)
MIN_PRESENCE = 0.5       # punkt z mniejszą pewnością jest ignorowany (kość zostaje jak była)
USE_HANDS = False         # dłonie i palce
HANDEDNESS_HIGH_IS_RIGHT = True   # tylko dla starych nagrań bez współrzędnych ekranowych dłoni
# ========================================================

with open(PATH, "r") as f:
    data = json.load(f)

frames = data["frames"]
fps = max(1, round(data.get("fps", 30)))
scene = bpy.context.scene
scene.render.fps = fps

arm = bpy.data.objects[ARMATURE_NAME]
bones = arm.data.bones
pbones = arm.pose.bones

SX = 1.0 if MIRROR else -1.0
# Obrót obiektu armatury (kierunki liczymy w jej przestrzeni)
W2A = arm.matrix_world.to_quaternion().to_matrix().inverted()


def has(short):
    return (PREFIX + short) in bones


def rest_pos(short):
    return bones[PREFIX + short].matrix_local.to_translation()


def P(p):
    """MediaPipe (x w prawo, y w dół, z do kamery=mniejsze) -> Blender (X, Y w głąb, Z w górę)."""
    return Vector((SX * p[0], p[2], -p[1]))


def lmi(left_idx, char_side):
    """Indeks punktu pozy dla danej strony postaci (z uwzględnieniem MIRROR)."""
    right_idx = left_idx + 1
    if char_side == "Left":
        return left_idx if MIRROR else right_idx
    return right_idx if MIRROR else left_idx


def triad(up, side):
    """Układ współrzędnych (macierz 3x3) z kierunku 'góra' i kierunku 'bok'."""
    z = up.normalized()
    x = side - side.dot(z) * z
    if x.length < 1e-6:
        return None
    x.normalize()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed()


# ---------------------------------------------------------------- szkielet
order = []


def _add(b):
    order.append(b)
    for c in b.children:
        _add(c)


for b in bones:
    if b.parent is None:
        _add(b)

REL = {
    b.name: (b.matrix_local if b.parent is None
             else b.parent.matrix_local.inverted() @ b.matrix_local)
    for b in order
}

# Kości sterowane "układem" (kierunek góra + bok): biodra i dłonie
TRIADS = {}


def _rest_triad(key, up_from, up_to, side_a, side_b):
    names = [key, up_from, up_to, side_a, side_b]
    if not all(has(n) for n in names):
        print("Brak kości dla", key, [n for n in names if not has(n)])
        return
    F = triad(rest_pos(up_to) - rest_pos(up_from), rest_pos(side_a) - rest_pos(side_b))
    if F is not None:
        TRIADS[key] = F


_rest_triad("Hips", "Hips", "Spine", "LeftUpLeg", "RightUpLeg")
for _s in ("Left", "Right"):
    _rest_triad(f"{_s}Hand", f"{_s}Hand", f"{_s}HandMiddle1",
                f"{_s}HandIndex1", f"{_s}HandPinky1")

# ---------------------------------------------------------------- stan (trzymanie pozycji)
state = {}   # klucz -> wygładzony kierunek (Vector w przestrzeni armatury)


def put(key, v):
    if v.length < 1e-6:
        return
    v = v.normalized()
    old = state.get(key)
    if old is None:
        state[key] = v
        return
    s = old.lerp(v, SMOOTHING)
    if s.length > 1e-6:
        state[key] = s.normalized()


def update_pose(pose):
    if pose is None:
        return  # brak wykrycia -> zostaje poprzednia pozycja

    def ok(*idx):
        return all(len(pose[i]) < 5 or pose[i][4] >= MIN_PRESENCE for i in idx)

    def pt(i):
        return P(pose[i])

    def vec(a, b):
        return W2A @ (pt(b) - pt(a))

    lh, rh = lmi(23, "Left"), lmi(23, "Right")
    ls, rs = lmi(11, "Left"), lmi(11, "Right")

    if ok(lh, rh, ls, rs):
        hip_mid = (pt(lh) + pt(rh)) / 2
        sh_mid = (pt(ls) + pt(rs)) / 2
        up = W2A @ (sh_mid - hip_mid)
        put("Hips_up", up)
        put("Hips_side", W2A @ (pt(lh) - pt(rh)))
        for name in ("Spine", "Spine1", "Spine2"):
            put(name, up)
        if ok(7, 8):
            head_dir = W2A @ (((pt(7) + pt(8)) / 2) - sh_mid)
            put("Neck", head_dir)
            put("Head", head_dir)

    limbs = [("Arm", 11, 13), ("ForeArm", 13, 15),
             ("UpLeg", 23, 25), ("Leg", 25, 27), ("Foot", 27, 31)]
    for side in ("Left", "Right"):
        for name, a, b in limbs:
            ia, ib = lmi(a, side), lmi(b, side)
            if ok(ia, ib):
                put(side + name, vec(ia, ib))


FINGERS = {"Thumb": 1, "Index": 5, "Middle": 9, "Ring": 13, "Pinky": 17}


def update_hands(fr): # Wyswietlanie dloni w zaleznosci od ustawien
    if not USE_HANDS:
        return
    hands = fr.get("hands") or []
    if not hands:
        return

    pose_s = fr.get("pose_screen")
    assigned = {}
    for h in hands:
        scr = h.get("screen")
        if pose_s is not None and scr is not None:
            w = scr[0]
            dl = math.hypot(w[0] - pose_s[15][0], w[1] - pose_s[15][1])
            dr = math.hypot(w[0] - pose_s[16][0], w[1] - pose_s[16][1])
            model_left = dl <= dr
            dist = min(dl, dr)
        else:
            high = h.get("handedness", 0.0) > 0.5
            model_left = (high != HANDEDNESS_HIGH_IS_RIGHT)
            dist = 0.0

        if MIRROR:
            char_side = "Left" if model_left else "Right"
        else:
            char_side = "Right" if model_left else "Left"

        if char_side not in assigned or dist < assigned[char_side][0]:
            assigned[char_side] = (dist, h)

    for side, (_, h) in assigned.items():
        w = h["world"]

        def v(a, b):
            return W2A @ (P(w[b]) - P(w[a]))

        put(f"{side}Hand_up", v(0, 9))
        put(f"{side}Hand_side", v(17, 5))
        for finger, base in FINGERS.items():
            for k in range(3):
                put(f"{side}Hand{finger}{k + 1}", v(base + k, base + k + 1))


# ---------------------------------------------------------------- rozwiązanie pozy
def solve():
    """Zwraca {nazwa_kości: quaternion rotacji (matrix_basis)} dla kości sterowanych."""
    M = {}
    out = {}
    for b in order:
        M0 = REL[b.name] if b.parent is None else M[b.parent.name] @ REL[b.name]
        short = b.name[len(PREFIX):] if b.name.startswith(PREFIX) else None
        R3 = M0.to_3x3()
        Mn = M0

        if short in TRIADS and (short + "_up") in state and (short + "_side") in state:
            Ft = triad(state[short + "_up"], state[short + "_side"])
            if Ft is not None:
                Q = Ft @ TRIADS[short].inverted()
                Mn = Matrix.Translation(M0.to_translation()) @ (Q @ R3).to_4x4()
        elif short in state:
            y = R3 @ Vector((0.0, 1.0, 0.0))
            rot = y.rotation_difference(state[short]).to_matrix()
            Mn = Matrix.Translation(M0.to_translation()) @ (rot @ R3).to_4x4()

        M[b.name] = Mn
        if Mn is not M0:
            out[b.name] = (M0.inverted() @ Mn).to_quaternion()
    return out


# ---------------------------------------------------------------- czyszczenie rigu
arm.animation_data_clear()   # usuń stare klatki kluczowe (np. z poprzedniego bake)
for pb in pbones:
    for c in list(pb.constraints):
        if c.type in {'DAMPED_TRACK', 'COPY_LOCATION'}:
            pb.constraints.remove(c)
    pb.rotation_mode = 'QUATERNION'
    pb.location = (0, 0, 0)
    pb.rotation_quaternion = (1, 0, 0, 0)
    pb.scale = (1, 1, 1)

# ---------------------------------------------------------------- główna pętla
last_frame = 1
for fr in frames:
    update_pose(fr.get("pose"))
    update_hands(fr)

    frame_no = int(round(fr["t"] * fps)) + 1
    last_frame = max(last_frame, frame_no)

    for name, q in solve().items():
        pb = pbones[name]
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=frame_no)

scene.frame_start = 1
scene.frame_end = last_frame
scene.frame_set(1)
print(f"Gotowe: {len(frames)} klatek nagrania, {fps} FPS, zakres 1-{last_frame}")