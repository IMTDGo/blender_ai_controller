# -*- coding: utf-8 -*-
"""
Blender AI Controller (Ollama) - Copilot-style framework
========================================================
Control Blender with natural language using a local Ollama model.
No Claude / cloud required.

Modes (decide whether the AI may use tools):
  - Ask   : no tools, only answers questions.
  - Plan  : no tools, guides you to build a todo plan, does not execute.
  - Agent : unlocks all tools and actually executes.

Permission (only effective in Agent mode):
  - read   : only plan / read the scene, no modification.
  - accept : ask before every modifying step (approve / skip / cancel).
  - auto   : run fully automatically; self-fix on failure, ask only when retries run out.

Per-step vision check (Agent, optional):
  - After each step, the AI picks a view angle (six faces / iso / free), a dedicated
    camera auto-frames the object just edited, and a screenshot is sent to a vision
    model (e.g. qwen3.6) to judge whether the step is correct.
  - If the AI thinks something is wrong, it pauses for you to "Fix" or "Accept & continue".

Aesthetics helpers:
  - "Beautify scene" sets EEVEE material preview + sky lighting + a sun + default materials.
  - Prefer-procedural prompt nudges Geometry Nodes + Shader nodes.
  - PolyHaven helpers (ph_hdri / ph_texture) fetch free HDRIs and PBR materials.

Safety:
  - The AI's generated Python runs via exec(). "Safe scan" (on by default) blocks code
    containing dangerous calls (file/system/network). You run AI-generated code at your
    own risk; review the report.

Interaction:
  - While the AI is running, inputs/settings are locked until it fully finishes.
  - A "Force stop" button can interrupt at any time.
  - Driven by bpy.app.timers on the main thread; background threads only call Ollama.
"""

bl_info = {
    "name": "Blender AI Controller (Ollama)",
    "author": "Z1jay",
    "version": (4, 2, 0),
    "blender": (4, 1, 0),
    "location": "View3D > Sidebar (N) > AI",
    "description": "Copilot-style local-AI control for Blender via Ollama (Ask/Plan/Agent, with per-step vision).",
    "category": "3D View",
}

import bpy
import json
import re
import threading
import queue
import urllib.request
import urllib.error
import traceback
import textwrap
import os
import tempfile
import base64
import math
import hashlib
import mathutils

from bpy.props import (
    StringProperty, BoolProperty, EnumProperty, IntProperty, CollectionProperty,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 180
WRAP = 46
MAX_REPORT = 250
TICK = 0.15  # state-machine poll interval (seconds)

RECOMMENDED = [
    "qwen2.5-coder:32b",
    "qwen3-coder:30b",
    "qwen2.5-coder:14b",
    "qwen3:32b",
    "qwen3.6:27b",
]

_MODEL_ITEMS = [("__none__", "(press Refresh to load models)", "")]

TODO_ICON = {
    "pending": 'DOT',
    "running": 'PLAY',
    "done": 'CHECKMARK',
    "skip": 'TRIA_RIGHT',
    "fail": 'ERROR',
}

AI_CAM_NAME = "AI_View_Cam"
VIEW_DIRS = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    "iso": (1.0, -1.0, 1.0),
}

# Safe-scan blacklist: substrings that are blocked in AI-generated step code.
UNSAFE_PATTERNS = [
    "import os", "import sys", "import subprocess", "import shutil", "import socket",
    "import requests", "import urllib", "from os", "from subprocess", "from shutil",
    "__import__", "subprocess", "os.system", "os.popen", "os.remove", "os.unlink",
    "os.rmdir", "shutil.rmtree", "shutil.move", "open(", "eval(", "exec(", "compile(",
    "bpy.ops.wm.read_homefile", "bpy.ops.wm.save", "bpy.ops.wm.quit",
    "bpy.ops.wm.open_mainfile", "bpy.ops.wm.read_factory",
]

# ---------------------------------------------------------------------------
# JSON schema
# ---------------------------------------------------------------------------
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "task_name": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "code": {"type": "string"},
                    "view": {"type": "string"},
                    "azimuth": {"type": "number"},
                    "elevation": {"type": "number"},
                    "target": {"type": "string"},
                },
                "required": ["name", "code"],
            },
        },
    },
    "required": ["task_name", "steps"],
}

FIX_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
}

VCHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "issue": {"type": "string"},
    },
    "required": ["ok", "issue"],
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
ASK_SYSTEM = """You are an assistant for Blender 4.x. Answer the user's questions and give advice in clear English.
This is "Ask mode": reply conversationally only. Do NOT output an executable step plan, and do NOT modify the scene.
If the user wants to actually create or modify something, suggest switching to Agent mode."""

PLAN_SYSTEM = """You are a Blender 4.1 Python (bpy) expert. The user describes a 3D task in natural language.
Break it into sequential steps and reply STRICTLY as a JSON object:
{"task_name": "short name", "steps": [{"name": "step description", "code": "bpy code", "view": "iso"}]}

Rules:
- For simple tasks use a single step; only split into multiple steps when the task is complex (your judgment).
- Each "code" must be complete, standalone Python runnable via exec(); `bpy` is already provided (do not import it).
- Use only Blender 4.x APIs; do not use 2.7x/2.8x style or invent parameters.
- After creating an object, get it via bpy.context.active_object before operating on it.
- Set material color via nodes: mat = bpy.data.materials.new("name"); mat.use_nodes = True;
  bsdf = mat.node_tree.nodes.get("Principled BSDF"); bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0);
  obj.data.materials.append(mat)
- Do not read/write files, run system commands, access the network, or save files.

Quality (make results look good, do not just stack boxes):
- Use modifiers where appropriate: BEVEL for rounded edges, SUBSURF for smoothing (and call bpy.ops.object.shade_smooth()),
  ARRAY for repetition, MIRROR for symmetry.
  e.g. m = obj.modifiers.new("Bevel", "BEVEL"); m.width = 0.02; m.segments = 3
  e.g. m = obj.modifiers.new("Subsurf", "SUBSURF"); m.levels = 2; bpy.ops.object.shade_smooth()
- Give every visible object a Principled BSDF material with sensible Base Color / Roughness / Metallic, not flat grey.
- Mind proportions and detail; add lighting when needed.

Each step may choose a view to inspect the result (the object just edited is auto-framed):
- "view": front / back / left / right / top / bottom / iso / free / auto (default iso).
- With "free", add "azimuth" (degrees) and "elevation" (degrees).
- "target": name of the object to observe (optional; defaults to the object just edited).

Example
User: create a red sphere and add a point light above it
Reply:
{"task_name": "Red sphere with point light", "steps": [
  {"name": "Create sphere", "code": "bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0))", "view": "front"},
  {"name": "Apply red material", "code": "obj = bpy.context.active_object\\nmat = bpy.data.materials.new('Red')\\nmat.use_nodes = True\\nbsdf = mat.node_tree.nodes.get('Principled BSDF')\\nbsdf.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)\\nobj.data.materials.append(mat)", "view": "iso"},
  {"name": "Add point light", "code": "bpy.ops.object.light_add(type='POINT', location=(0, 0, 3))", "view": "front"}
]}
"""

PLAN_EXTRA = "\nNote: this is Plan mode. Split into clear multi-step todos so the user can review each item."

PROC_EXTRA = """

Prefer a procedural approach and combine Geometry Nodes with Shader nodes:
- Geometry: when suitable, use Geometry Nodes for procedural geometry (scatter, instancing, arrays, deformation, detail) instead of hand-placing vertices.
- Material: use Shader nodes for procedural materials (noise/voronoi/color ramp -> Base Color / Roughness / normal bump), not just a single color.
- Combine: e.g. generate/scatter geometry with GN, then give it a procedural shader material.

Correct way to build Geometry Nodes in Blender 4.1 (4.x uses node_group.interface, NOT the old inputs/outputs):
obj = bpy.context.active_object
mod = obj.modifiers.new("GeoNodes", 'NODES')
ng = bpy.data.node_groups.new("MyGeo", 'GeometryNodeTree')
mod.node_group = ng
ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
gin = ng.nodes.new("NodeGroupInput"); gin.location = (-400, 0)
gout = ng.nodes.new("NodeGroupOutput"); gout.location = (400, 0)
# put processing nodes in between, e.g. GeometryNodeDistributePointsOnFaces / GeometryNodeInstanceOnPoints / GeometryNodeSetPosition
ng.links.new(gin.outputs["Geometry"], gout.inputs["Geometry"])

Procedural shader material skeleton:
mat = bpy.data.materials.new("Proc"); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes.get("Principled BSDF")
noise = nt.nodes.new("ShaderNodeTexNoise")
ramp = nt.nodes.new("ShaderNodeValToRGB")
nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
obj.data.materials.append(mat)
"""

PH_HELP = """

You may use free PolyHaven assets (built-in helpers, call them directly in code, no import needed):
- ph_hdri("keyword", res="2k"): download an HDRI and set it as the world environment light. e.g. ph_hdri("sunset"), ph_hdri("studio"), ph_hdri("forest").
- ph_texture("keyword", res="2k"): download PBR texture maps, build a material, and apply it to the active object. e.g. ph_texture("wood floor"), ph_texture("rusty metal"), ph_texture("bricks").
Use these for realistic lighting and materials; far better than flat colors. Note: downloading needs network and takes a few seconds; call once per object that needs a material.
"""

FIX_SYSTEM = """You are a Blender 4.1 Python (bpy) debugging expert.
You receive failing bpy code, an error message (or a visual issue), and the current scene object list.
Fix it so it runs correctly in Blender 4.1 and matches the intent. Use only 4.x APIs.
Reply STRICTLY as JSON: {"code": "corrected, directly runnable bpy code"}"""

SUMMARY_SYSTEM = "You are a Blender assistant. Reply concisely in English."

VCHECK_SYSTEM = """You are a Blender assistant that looks at a 3D viewport screenshot and judges whether the step just completed is correct and matches the intent.
Reply STRICTLY as JSON: {"ok": true or false, "issue": "short English description if there is a problem, empty string otherwise"}.
Only return false when clearly wrong (e.g. an expected object is missing, material clearly wrong, position clearly off)."""


# ---------------------------------------------------------------------------
# Text / JSON helpers
# ---------------------------------------------------------------------------
def _strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def clean_code(text):
    text = _strip_think(text)
    if "```" in text:
        blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.S)
        if blocks:
            text = max(blocks, key=len)
    return text.strip()


def parse_json(text):
    text = _strip_think(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def check_unsafe(code):
    low = code.replace(" ", "").lower()
    for pat in UNSAFE_PATTERNS:
        if pat.replace(" ", "").lower() in low:
            return pat
    return None


# ---------------------------------------------------------------------------
# Ollama calls (used only on background threads; never touch bpy)
# ---------------------------------------------------------------------------
def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + path, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_chat(model, messages, fmt=None, temperature=0.2):
    base = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if fmt is not None:
        base["format"] = fmt
    try:
        payload = dict(base)
        payload["think"] = False
        resp = _post("/api/chat", payload)
    except urllib.error.HTTPError:
        resp = _post("/api/chat", base)
    return resp.get("message", {}).get("content", "")


def list_models():
    req = urllib.request.Request(OLLAMA_URL + "/api/tags")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", [])]


def list_running():
    req = urllib.request.Request(OLLAMA_URL + "/api/ps")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    out = []
    for m in data.get("models", []):
        out.append((m.get("name", ""), m.get("size_vram", m.get("size", 0)) or 0))
    return out


def loaded_text():
    try:
        running = list_running()
    except Exception:
        return "(query failed)"
    if not running:
        return "none"
    return ", ".join("%s (%.1f GB)" % (n, v / 1e9) for n, v in running)


def unload_model(name):
    _post("/api/generate", {"model": name, "keep_alive": 0})


def get_capabilities(model):
    """Query a model's capabilities via /api/show (e.g. completion, tools, vision, thinking)."""
    try:
        resp = _post("/api/show", {"model": model})
        caps = resp.get("capabilities") or []
        return [str(c) for c in caps]
    except Exception:
        return []


def refresh_caps(st):
    """Read the selected model's capabilities into the panel state. Main thread only."""
    m = getattr(st, "model", "")
    if not m or m == "__none__":
        st.caps = ""
        st.caps_known = False
        st.model_vision = False
        return
    caps = get_capabilities(m)
    st.caps = ", ".join(caps) if caps else "(unknown)"
    st.caps_known = bool(caps)
    st.model_vision = ("vision" in caps)


def _on_model_change(self, context):
    try:
        refresh_caps(self)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Local memory / RAG (keyword retrieval; pure Python, no extra deps)
#   - Seed KB shipped in kb/seed.jsonl
#   - User episodic memory appended to the addon config dir
#   - Preference notes live in AddonPreferences (always injected)
# ---------------------------------------------------------------------------
_LAST_RUN = {"prompt": "", "code": "", "mode": ""}
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[一-鿿]")


def _mem_dir():
    try:
        return bpy.utils.user_resource('CONFIG', path="blender_ai_controller", create=True)
    except Exception:
        return tempfile.gettempdir()


def _episodes_path():
    return os.path.join(_mem_dir(), "episodes.jsonl")


def _tokens(s):
    return set(w.lower() for w in _WORD_RE.findall(s or ""))


def _load_memory_items():
    items = []
    seed = os.path.join(os.path.dirname(__file__), "kb", "seed.jsonl")
    for path in (seed, _episodes_path()):
        try:
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            items.append(json.loads(line))
                        except Exception:
                            pass
        except Exception:
            pass
    return items


def retrieve_memory_keyword(query, k=3):
    qt = _tokens(query)
    if not qt:
        return []
    scored = []
    for it in _load_memory_items():
        pt = _tokens(it.get("prompt", ""))
        if not pt:
            continue
        overlap = len(qt & pt)
        if overlap:
            scored.append((overlap / (len(pt) ** 0.5), it))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in scored[:k]]


def record_episode(prompt, code, mode, source="auto"):
    if not (prompt and code):
        return
    try:
        with open(_episodes_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "code": code, "mode": mode,
                                "source": source}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def count_episodes():
    try:
        p = _episodes_path()
        if not os.path.exists(p):
            return 0
        with open(p, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def build_examples_context(user_input):
    eps = retrieve_memory(user_input, k=3)
    if not eps:
        return ""
    ex = "\n\n".join("# Past task: %s\n%s" % (e.get("prompt", ""), e.get("code", ""))
                     for e in eps)
    return ("Relevant examples that worked before on this machine "
            "(reuse patterns that fit, adapt as needed):\n" + ex)


def _get_prefs():
    try:
        return bpy.context.preferences.addons[__name__].preferences
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Optional semantic retrieval via an Ollama embedding model.
# Vector cache lives next to episodes.jsonl, keyed by embedding-model name.
# ---------------------------------------------------------------------------
def embed_text(model, text):
    try:
        resp = _post("/api/embed", {"model": model, "input": text})
        embs = resp.get("embeddings")
        if embs:
            return embs[0]
    except Exception:
        pass
    try:
        resp = _post("/api/embeddings", {"model": model, "prompt": text})
        e = resp.get("embedding")
        if e:
            return e
    except Exception:
        pass
    return None


def _emb_cache_path(model):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model or "model")
    return os.path.join(_mem_dir(), "emb_%s.json" % safe)


def _load_emb_cache(model):
    p = _emb_cache_path(model)
    try:
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_emb_cache(model, cache):
    try:
        with open(_emb_cache_path(model), "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _hash(t):
    return hashlib.sha1((t or "").encode("utf-8")).hexdigest()


def retrieve_memory_semantic(query, model, k=3):
    try:
        import numpy as np
    except Exception:
        return []
    items = _load_memory_items()
    if not items:
        return []
    cache = _load_emb_cache(model)
    dirty = False
    vecs = []
    kept = []
    for it in items:
        p = it.get("prompt", "")
        if not p:
            continue
        h = _hash(p)
        v = cache.get(h)
        if v is None:
            v = embed_text(model, p)
            if v is None:
                continue
            cache[h] = v
            dirty = True
        vecs.append(v)
        kept.append(it)
    if dirty:
        _save_emb_cache(model, cache)
    if not kept:
        return []
    qv = embed_text(model, query)
    if qv is None:
        return []
    try:
        M = np.array(vecs, dtype=float)
        q = np.array(qv, dtype=float)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        qn = q / (np.linalg.norm(q) + 1e-9)
        sims = Mn @ qn
        idx = list(np.argsort(-sims)[:k])
        return [kept[i] for i in idx]
    except Exception:
        return []


def retrieve_memory(query, k=3):
    """Dispatch: semantic if an embedding model is configured & works, else keyword."""
    prefs = _get_prefs()
    model = getattr(prefs, "embed_model", "none") if prefs else "none"
    if model and model not in ("none", ""):
        try:
            res = retrieve_memory_semantic(query, model, k)
            if res:
                return res
        except Exception:
            pass
    return retrieve_memory_keyword(query, k)


# Background pull of an embedding model via Ollama /api/pull
_PULL = {"active": False, "status": "", "model": ""}


def _pull_worker(model):
    _PULL.update(active=True, status="starting...", model=model)
    try:
        data = json.dumps({"model": model, "stream": True}).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL + "/api/pull", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    o = json.loads(raw)
                except Exception:
                    continue
                if o.get("error"):
                    _PULL["status"] = "error: " + str(o["error"])
                    continue
                stt = o.get("status", "")
                if o.get("total") and o.get("completed") is not None:
                    pct = int(o["completed"] * 100 / max(o["total"], 1))
                    _PULL["status"] = "%s %d%%" % (stt, pct)
                elif stt:
                    _PULL["status"] = stt
        if not _PULL["status"].startswith("error"):
            _PULL["status"] = "done"
    except Exception as e:
        _PULL["status"] = "error: %s" % e
    finally:
        _PULL["active"] = False


def _pull_tick():
    _redraw()
    return 0.5 if _PULL["active"] else None


# ---------------------------------------------------------------------------
# Run step code on the main thread
# ---------------------------------------------------------------------------
def run_step_code(code, safe_scan=True):
    code = clean_code(code)
    if not code.strip():
        return False, "(empty code)"
    if safe_scan:
        bad = check_unsafe(code)
        if bad:
            return False, "Safe scan blocked: detected '%s' (disable Safe scan to allow)" % bad
    try:
        ns = {"bpy": bpy, "__name__": "__ai_step__",
              "ph_hdri": ph_hdri, "ph_texture": ph_texture}
        exec(compile(code, "<ai_step>", "exec"), ns)
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def scene_snapshot():
    objs = ["%s(%s)" % (o.name, o.type) for o in bpy.data.objects]
    if not objs:
        return "Scene objects: (empty)"
    return "Scene objects: " + ", ".join(objs[:60])


# ---------------------------------------------------------------------------
# View / screenshot (main thread only)
# ---------------------------------------------------------------------------
def _view_dir(view, az, el):
    if view in VIEW_DIRS:
        return mathutils.Vector(VIEW_DIRS[view]).normalized()
    if view == "free":
        a = math.radians(az)
        e = math.radians(el)
        return mathutils.Vector((math.cos(e) * math.sin(a),
                                 -math.cos(e) * math.cos(a),
                                 math.sin(e))).normalized()
    return mathutils.Vector(VIEW_DIRS["iso"]).normalized()


def _bounds(objs):
    pts = []
    for o in objs:
        if o.type in {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'} and hasattr(o, "bound_box"):
            for c in o.bound_box:
                pts.append(o.matrix_world @ mathutils.Vector(c))
        else:
            pts.append(o.matrix_world.translation.copy())
    if not pts:
        return mathutils.Vector((0.0, 0.0, 0.0)), 1.0
    minc = mathutils.Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    maxc = mathutils.Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    center = (minc + maxc) / 2.0
    radius = max((maxc - center).length, 0.5)
    return center, radius


def _ensure_ai_cam():
    cam = bpy.data.objects.get(AI_CAM_NAME)
    if cam is None or cam.type != 'CAMERA':
        cam_data = bpy.data.cameras.new(AI_CAM_NAME)
        cam = bpy.data.objects.new(AI_CAM_NAME, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    cam.hide_viewport = True
    cam.hide_select = True
    return cam


def capture_view(view="iso", azimuth=45.0, elevation=30.0, target_name=""):
    """Set up a view (AI-chosen direction, auto-framed) and take a screenshot.
    Returns a PNG path or None. Main thread only."""
    scn = bpy.context.scene
    r = scn.render
    win = bpy.context.window
    area = region = None
    if win and win.screen:
        for a in win.screen.areas:
            if a.type == 'VIEW_3D':
                area = a
                region = next((rg for rg in a.regions if rg.type == 'WINDOW'), None)
                break
    if not area or not region:
        return None

    prev_sel = [o for o in bpy.context.selected_objects]
    prev_active = bpy.context.view_layer.objects.active
    saved_cam = scn.camera
    saved_path = r.filepath
    saved_fmt = r.image_settings.file_format
    try:
        r.image_settings.file_format = 'PNG'
        r.filepath = os.path.join(tempfile.gettempdir(), "blender_ai_view_")

        tgt = bpy.data.objects.get(target_name) if target_name else None
        if tgt is not None:
            objs = [tgt]
        else:
            objs = [o for o in bpy.context.selected_objects] or list(scn.objects)
        objs = [o for o in objs if o.name != AI_CAM_NAME] or list(scn.objects)

        if view == "auto" and saved_cam is not None:
            pass  # use the user's existing camera
        else:
            cam = _ensure_ai_cam()
            center, radius = _bounds(objs)
            d = _view_dir(view, azimuth, elevation)
            cam.location = center + d * (radius * 3.0 + 1.0)
            look = center - cam.location
            if look.length > 1e-6:
                cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
            scn.camera = cam

        for o in bpy.context.selected_objects:
            o.select_set(False)
        framed = False
        for o in objs:
            try:
                o.select_set(True)
                framed = True
            except Exception:
                pass
        if objs:
            bpy.context.view_layer.objects.active = objs[0]

        with bpy.context.temp_override(window=win, area=area, region=region):
            if framed:
                try:
                    bpy.ops.view3d.camera_to_view_selected()
                except Exception:
                    pass
            bpy.ops.render.opengl(write_still=True)
        return scn.render.frame_path()
    except Exception:
        return None
    finally:
        try:
            scn.camera = saved_cam
            for o in bpy.context.selected_objects:
                o.select_set(False)
            for o in prev_sel:
                try:
                    o.select_set(True)
                except Exception:
                    pass
            bpy.context.view_layer.objects.active = prev_active
        except Exception:
            pass
        r.filepath = saved_path
        r.image_settings.file_format = saved_fmt


# ---------------------------------------------------------------------------
# Scene beautify (lighting / materials / render; self-contained, no download)
# ---------------------------------------------------------------------------
def setup_sky_world(scn):
    world = scn.world
    if world is None:
        world = bpy.data.worlds.new("AI_World")
        scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs[1].default_value = 1.0
    try:
        sky = nt.nodes.new("ShaderNodeTexSky")
        sky.sky_type = 'NISHITA'
        nt.links.new(sky.outputs[0], bg.inputs[0])
    except Exception:
        bg.inputs[0].default_value = (0.05, 0.05, 0.06, 1.0)
    nt.links.new(bg.outputs[0], out.inputs[0])


def set_viewport_material():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'MATERIAL'
    except Exception:
        pass


def beautify_scene():
    """Switch to EEVEE material preview, add sky lighting + a sun, give bare meshes a PBR material."""
    scn = bpy.context.scene
    for eng in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
        try:
            scn.render.engine = eng
            break
        except Exception:
            continue
    setup_sky_world(scn)
    if not any(o.type == 'LIGHT' for o in scn.objects):
        try:
            bpy.ops.object.light_add(type='SUN', location=(0.0, 0.0, 10.0))
            sun = bpy.context.active_object
            sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(60))
            sun.data.energy = 3.0
        except Exception:
            pass
    for o in scn.objects:
        if o.type == 'MESH' and o.data is not None and not o.data.materials:
            m = bpy.data.materials.new(o.name + "_mat")
            m.use_nodes = True
            b = m.node_tree.nodes.get("Principled BSDF")
            if b:
                try:
                    b.inputs["Roughness"].default_value = 0.5
                except Exception:
                    pass
            o.data.materials.append(m)
    set_viewport_material()


# ---------------------------------------------------------------------------
# PolyHaven free assets (HDRI / PBR materials)
# These helpers are exposed in the step namespace so the AI can call them.
# Note: downloading uses the network and briefly blocks the main thread.
# ---------------------------------------------------------------------------
PH_API = "https://api.polyhaven.com"


def _ph_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "BlenderAIController"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _ph_cache_dir():
    d = os.path.join(tempfile.gettempdir(), "blender_ai_polyhaven")
    os.makedirs(d, exist_ok=True)
    return d


def _ph_download(url, filename):
    dest = os.path.join(_ph_cache_dir(), filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "BlenderAIController"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest


def _ph_pick(asset_type, keyword):
    data = _ph_get_json("%s/assets?type=%s" % (PH_API, asset_type))
    if not isinstance(data, dict) or not data:
        return None
    kw = (keyword or "").lower().strip()
    if not kw:
        return next(iter(data))
    best = None
    best_score = 0
    for slug, info in data.items():
        hay = " ".join([slug, info.get("name", "")] + list(info.get("tags", []))
                       + list(info.get("categories", []))).lower()
        score = 0
        if kw in slug.lower():
            score += 3
        if kw in info.get("name", "").lower():
            score += 2
        for t in kw.split():
            if t in hay:
                score += 1
        if score > best_score:
            best_score = score
            best = slug
    return best or next(iter(data))


def _ph_res(entry, res):
    return entry.get(res) or entry.get("2k") or (next(iter(entry.values())) if entry else None)


def ph_hdri(keyword="", res="2k"):
    """Download a PolyHaven HDRI and set it as the world environment light."""
    slug = _ph_pick("hdris", keyword)
    if not slug:
        print("PolyHaven: no HDRI found"); return None
    files = _ph_get_json("%s/files/%s" % (PH_API, slug))
    resd = _ph_res(files.get("hdri", {}), res)
    if not resd:
        print("PolyHaven: HDRI has no files"); return None
    fmt = resd.get("hdr") or resd.get("exr") or next(iter(resd.values()))
    url = fmt.get("url")
    path = _ph_download(url, slug + "_" + res + os.path.splitext(url)[1])
    img = bpy.data.images.load(path, check_existing=True)
    scn = bpy.context.scene
    world = scn.world or bpy.data.worlds.new("World")
    scn.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = img
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    print("PolyHaven: applied HDRI", slug)
    return slug


def ph_texture(keyword="", res="2k", assign=True):
    """Download PolyHaven PBR maps, build a material, apply it to the active object. Returns the material."""
    slug = _ph_pick("textures", keyword)
    if not slug:
        print("PolyHaven: no texture found"); return None
    files = _ph_get_json("%s/files/%s" % (PH_API, slug))

    def grab(*subs):
        for k in files:
            kl = k.lower()
            if any(s in kl for s in subs):
                resd = _ph_res(files.get(k, {}), res)
                if not resd:
                    continue
                fmt = resd.get("jpg") or resd.get("png") or next(iter(resd.values()))
                if isinstance(fmt, dict) and fmt.get("url"):
                    return fmt["url"]
        return None

    col_url = grab("diff", "albedo")
    rough_url = grab("rough")
    nor_url = grab("nor_gl", "nor_dx", "normal", "nor")
    mat = bpy.data.materials.new("PH_" + slug)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if col_url:
        p = _ph_download(col_url, slug + "_col" + os.path.splitext(col_url)[1])
        img = bpy.data.images.load(p, check_existing=True)
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        if bsdf:
            nt.links.new(n.outputs["Color"], bsdf.inputs["Base Color"])
    if rough_url and bsdf:
        p = _ph_download(rough_url, slug + "_rough" + os.path.splitext(rough_url)[1])
        img = bpy.data.images.load(p, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        nt.links.new(n.outputs["Color"], bsdf.inputs["Roughness"])
    if nor_url and bsdf:
        p = _ph_download(nor_url, slug + "_nor" + os.path.splitext(nor_url)[1])
        img = bpy.data.images.load(p, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = img
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if assign:
        obj = bpy.context.active_object
        if obj and obj.type == 'MESH':
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
    print("PolyHaven: applied texture", slug)
    return mat


# ---------------------------------------------------------------------------
# Report / todo list (main thread only)
# ---------------------------------------------------------------------------
def report_add(st, text):
    for raw in str(text).split("\n"):
        if raw == "":
            it = st.report.add()
            it.text = ""
            continue
        for line in textwrap.wrap(raw, WRAP) or [""]:
            it = st.report.add()
            it.text = line
    while len(st.report) > MAX_REPORT:
        st.report.remove(0)
    st.report_index = len(st.report) - 1


def todo_set(st, names):
    st.todo.clear()
    for n in names:
        it = st.todo.add()
        it.name = n
        it.status = "pending"
    st.todo_index = 0


def todo_status(st, idx, status):
    if 0 <= idx < len(st.todo):
        st.todo[idx].status = status
        st.todo_index = idx


# ---------------------------------------------------------------------------
# Runtime (single-task state; background work never touches bpy)
# ---------------------------------------------------------------------------
class Runtime:
    def __init__(self):
        self.q = queue.Queue()
        self.reset()

    def reset(self, model="", user_input="", mode="AGENT", permission="accept",
              auto_retry=2, scene_snap="", step_vision=False, prefer_nodes=True,
              use_polyhaven=True, safe_scan=True, memory_ctx="", extra_system=""):
        self.model = model
        self.memory_ctx = memory_ctx
        self.extra_system = extra_system
        self.user_input = user_input
        self.mode = mode
        self.permission = permission
        self.auto_retry = auto_retry
        self.scene_snap = scene_snap
        self.step_vision = step_vision
        self.prefer_nodes = prefer_nodes
        self.use_polyhaven = use_polyhaven
        self.safe_scan = safe_scan
        self.steps = []
        self.idx = 0
        self.retries = 0
        self.results = []
        self.last_error = ""
        self.phase = "IDLE"
        self.show_todo = False
        self.stop = False
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def poll(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def start_ask(self):
        threading.Thread(target=self._ask_worker, daemon=True).start()

    def _ask_worker(self):
        try:
            sysmsg = ASK_SYSTEM + (("\n\n" + self.extra_system) if self.extra_system else "")
            content = ollama_chat(self.model, [
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": self.user_input},
            ], temperature=0.4)
            self.q.put(("ok", _strip_think(content)))
        except Exception as e:
            self.q.put(("error", "%s" % e))

    def start_plan(self, include_scene):
        threading.Thread(target=self._plan_worker, args=(include_scene,), daemon=True).start()

    def _plan_worker(self, include_scene):
        try:
            sys_prompt = (PLAN_SYSTEM
                          + (PLAN_EXTRA if self.mode == "PLAN" else "")
                          + (PROC_EXTRA if self.prefer_nodes else "")
                          + (PH_HELP if self.use_polyhaven else "")
                          + (("\n\n" + self.extra_system) if self.extra_system else ""))
            umsg = self.user_input
            if include_scene and self.scene_snap:
                umsg = umsg + "\n\n" + self.scene_snap
            if self.memory_ctx:
                umsg = self.memory_ctx + "\n\n---\n\n" + umsg
            content = ollama_chat(self.model, [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": umsg},
            ], fmt=PLAN_SCHEMA)
            plan = parse_json(content)
            if not isinstance(plan, dict) or "steps" not in plan:
                self.q.put(("error", "Model output could not be parsed as a plan JSON"))
            else:
                self.q.put(("ok", plan))
        except Exception as e:
            self.q.put(("error", "%s" % e))

    def start_fix(self, step, err, snapshot):
        threading.Thread(target=self._fix_worker,
                         args=(dict(step), err, snapshot), daemon=True).start()

    def _fix_worker(self, step, err, snapshot):
        try:
            user = ("Original request: %s\nStep: %s\nThis bpy code has a problem:\n%s\n\nProblem:\n%s\n\n%s\n\nReturn only the corrected, directly runnable bpy code."
                    % (self.user_input, step.get("name", ""), step.get("code", ""),
                       err[-1500:], snapshot))
            content = ollama_chat(self.model, [
                {"role": "system", "content": FIX_SYSTEM},
                {"role": "user", "content": user},
            ], fmt=FIX_SCHEMA)
            obj = parse_json(content)
            if isinstance(obj, dict) and obj.get("code"):
                self.q.put(("ok", obj["code"]))
            else:
                self.q.put(("ok", clean_code(content)))
        except Exception as e:
            self.q.put(("error", "%s" % e))

    def start_summary(self, snapshot):
        detail = "\n".join("- %s: %s" % (n, s) for n, s, _ in self.results)
        threading.Thread(target=self._summary_worker,
                         args=(detail, snapshot), daemon=True).start()

    def _summary_worker(self, detail, snapshot):
        try:
            user = ("User request: %s\n\nActual results:\n%s\n\n%s\n\nIn English, summarize in <= 3 sentences what was done and whether any step failed or was skipped."
                    % (self.user_input, detail, snapshot))
            content = ollama_chat(self.model, [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": user},
            ], temperature=0.4)
            self.q.put(("ok", _strip_think(content)[:600]))
        except Exception as e:
            self.q.put(("error", "%s" % e))

    def start_vcheck(self, image_path, step_name):
        threading.Thread(target=self._vcheck_worker,
                         args=(image_path, step_name), daemon=True).start()

    def _vcheck_worker(self, image_path, step_name):
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            user = ("Goal: %s\nStep just completed: %s\nThis is the 3D viewport after the step. Judge whether the step was done correctly."
                    % (self.user_input, step_name))
            messages = [
                {"role": "system", "content": VCHECK_SYSTEM},
                {"role": "user", "content": user, "images": [b64]},
            ]
            content = ollama_chat(self.model, messages, fmt=VCHECK_SCHEMA, temperature=0.2)
            obj = parse_json(content)
            if isinstance(obj, dict) and "ok" in obj:
                self.q.put(("ok", (bool(obj.get("ok")), str(obj.get("issue", "")))))
            else:
                self.q.put(("ok", (True, "")))  # parse failure -> treat as pass, don't block
        except Exception as e:
            self.q.put(("error", "%s" % e))


R = Runtime()


# ---------------------------------------------------------------------------
# State machine (driven by bpy.app.timers on the main thread)
# ---------------------------------------------------------------------------
def _redraw():
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
    except Exception:
        pass


def _finish_run(st, status):
    st.is_running = False
    st.needs_decision = False
    st.needs_approve = False
    st.needs_vdecision = False
    st.decision = "NONE"
    st.status = status
    R.phase = "IDLE"
    R.stop = False
    try:
        st.loaded_info = loaded_text()
    except Exception:
        pass
    _redraw()


def _advance(st):
    R.idx += 1
    R.retries = 0
    if R.idx >= len(R.steps):
        R.phase = "SUMMARY_WAIT"
        st.status = "Summarizing..."
        R.start_summary(scene_snapshot())
    else:
        R.phase = "STEP_START"


def _step_machine(st):
    """Return True to keep polling; False means finished (already called _finish_run)."""
    ph = R.phase

    if ph == "ASK_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        if kind == "error":
            report_add(st, "ERROR " + payload)
            _finish_run(st, "Failed")
            return False
        report_add(st, "AI: " + payload)
        _finish_run(st, "Answer ready")
        return False

    if ph == "PLANSHOW_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        if kind == "error":
            report_add(st, "ERROR " + payload)
            _finish_run(st, "Planning failed")
            return False
        steps = payload.get("steps", [])
        st.task_name = payload.get("task_name", "")
        if not steps:
            _finish_run(st, "No steps")
            return False
        todo_set(st, [s.get("name", "") for s in steps])
        report_add(st, "Plan: %s" % st.task_name)
        for i, s in enumerate(steps):
            report_add(st, "  %d. %s" % (i + 1, s.get("name", "")))
        if R.mode == "AGENT":
            report_add(st, "(read permission: plan only, not executed. Switch to accept/auto to run.)")
        else:
            report_add(st, "(Plan mode: not executed. Switch to Agent to run.)")
        _finish_run(st, "Plan ready (not executed)")
        return False

    if ph == "PLAN_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        if kind == "error":
            report_add(st, "ERROR " + payload)
            _finish_run(st, "Planning failed")
            return False
        R.steps = payload.get("steps", [])
        st.task_name = payload.get("task_name", "")
        if not R.steps:
            _finish_run(st, "No steps")
            return False
        R.idx = 0
        R.retries = 0
        R.show_todo = len(R.steps) > 1
        if R.show_todo:
            todo_set(st, [s.get("name", "") for s in R.steps])
        report_add(st, "Plan: %s (%d steps)" % (st.task_name, len(R.steps)))
        R.phase = "STEP_START"
        return True

    if ph == "STEP_START":
        step = R.steps[R.idx]
        if R.show_todo:
            todo_status(st, R.idx, "running")
        st.status = "[%d/%d] %s" % (R.idx + 1, len(R.steps), step.get("name", ""))
        if R.permission == "accept":
            st.pending_name = step.get("name", "")
            st.pending_code = clean_code(step.get("code", ""))
            st.needs_approve = True
            R.phase = "APPROVE_WAIT"
        else:
            R.phase = "STEP_RUN"
        return True

    if ph == "APPROVE_WAIT":
        d = st.decision
        if d == "APPROVE":
            st.decision = "NONE"
            st.needs_approve = False
            R.phase = "STEP_RUN"
        elif d == "SKIP":
            st.decision = "NONE"
            st.needs_approve = False
            R.results.append((R.steps[R.idx].get("name", ""), "skipped", ""))
            if R.show_todo:
                todo_status(st, R.idx, "skip")
            report_add(st, "Skipped: %s" % R.steps[R.idx].get("name", ""))
            _advance(st)
        elif d == "CANCEL":
            st.decision = "NONE"
            st.needs_approve = False
            _finish_run(st, "Cancelled")
            return False
        return True

    if ph == "STEP_RUN":
        step = R.steps[R.idx]
        ok, err = run_step_code(step.get("code", ""), R.safe_scan)
        if ok:
            R.results.append((step.get("name", ""), "done", ""))
            if R.show_todo:
                todo_status(st, R.idx, "done")
            report_add(st, "OK %s" % step.get("name", ""))
            if R.step_vision:
                R.phase = "VCHECK_CAP"
            else:
                _advance(st)
            return True
        R.last_error = err
        last = err.strip().splitlines()[-1] if err.strip() else err
        if R.retries < R.auto_retry:
            R.retries += 1
            report_add(st, "WARN %s failed, AI fixing (%d/%d)" % (step.get("name", ""), R.retries, R.auto_retry))
            st.status = "Fixing..."
            R.phase = "FIX_WAIT"
            R.start_fix(step, err, scene_snapshot())
        else:
            if R.show_todo:
                todo_status(st, R.idx, "fail")
            report_add(st, "FAIL %s repeatedly: %s" % (step.get("name", ""), last[:80]))
            st.status = "Failed repeatedly; choose skip or cancel"
            st.needs_decision = True
            R.phase = "USER_WAIT"
        return True

    if ph == "FIX_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        if kind == "error":
            if R.show_todo:
                todo_status(st, R.idx, "fail")
            st.status = "Fix failed; choose skip or cancel"
            st.needs_decision = True
            R.phase = "USER_WAIT"
            return True
        R.steps[R.idx]["code"] = payload
        R.phase = "STEP_START" if R.permission == "accept" else "STEP_RUN"
        return True

    if ph == "USER_WAIT":
        d = st.decision
        if d == "SKIP":
            st.decision = "NONE"
            st.needs_decision = False
            R.results.append((R.steps[R.idx].get("name", ""), "skipped", R.last_error[-80:]))
            if R.show_todo:
                todo_status(st, R.idx, "skip")
            report_add(st, "Skipped: %s" % R.steps[R.idx].get("name", ""))
            _advance(st)
            return True
        if d == "CANCEL":
            st.decision = "NONE"
            st.needs_decision = False
            _finish_run(st, "Cancelled")
            return False
        return True

    if ph == "VCHECK_CAP":
        step = R.steps[R.idx]
        view = (step.get("view") or "iso").lower()
        if view not in VIEW_DIRS and view not in ("free", "auto"):
            view = "iso"
        try:
            az = float(step.get("azimuth", 45) or 45)
            el = float(step.get("elevation", 30) or 30)
        except Exception:
            az, el = 45.0, 30.0
        st.status = "Looking... (%s)" % view
        path = capture_view(view, az, el, step.get("target", "") or "")
        if not path:
            report_add(st, "Could not capture view; skipping visual check")
            _advance(st)
            return True
        R.start_vcheck(path, step.get("name", ""))
        R.phase = "VCHECK_WAIT"
        return True

    if ph == "VCHECK_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        if kind == "error":
            report_add(st, "Visual check failed; continuing")
            _advance(st)
            return True
        ok_flag, issue = payload
        if ok_flag:
            report_add(st, "Step %d looks OK" % (R.idx + 1))
            _advance(st)
        else:
            st.vissue = issue or "(no detail)"
            report_add(st, "Step %d may have an issue: %s" % (R.idx + 1, st.vissue))
            st.needs_vdecision = True
            st.status = "AI sees an issue; choose Fix or Continue"
            R.phase = "VCHECK_DECIDE"
        return True

    if ph == "VCHECK_DECIDE":
        d = st.decision
        if d == "VFIX":
            st.decision = "NONE"
            st.needs_vdecision = False
            report_add(st, "Fixing step %d for the visual issue..." % (R.idx + 1))
            st.status = "Fixing..."
            R.start_fix(R.steps[R.idx], "Visual check found a problem: " + st.vissue, scene_snapshot())
            R.phase = "FIX_WAIT"
        elif d == "VCONT":
            st.decision = "NONE"
            st.needs_vdecision = False
            report_add(st, "Accepted step %d, continuing" % (R.idx + 1))
            _advance(st)
        return True

    if ph == "SUMMARY_WAIT":
        msg = R.poll()
        if msg is None:
            return True
        kind, payload = msg
        report_add(st, "Summary: " + (payload if kind == "ok" else "(summary failed)"))
        try:
            code = "\n\n".join(s.get("code", "") for s in R.steps)
            _LAST_RUN["prompt"] = R.user_input
            _LAST_RUN["code"] = code
            _LAST_RUN["mode"] = R.mode
            if R.mode == "AGENT" and code.strip():
                record_episode(R.user_input, code, R.mode, "auto")
        except Exception:
            pass
        _finish_run(st, "Done")
        return False

    _finish_run(st, "(ended)")
    return False


def _tick():
    """bpy.app.timers callback: return the next interval (seconds) or None to stop."""
    try:
        st = bpy.context.scene.ai_props
    except Exception:
        return None
    try:
        if R.stop:
            report_add(st, "Force stopped")
            _finish_run(st, "Force stopped")
            return None
        cont = _step_machine(st)
    except Exception:
        try:
            report_add(st, "Internal error:\n" + traceback.format_exc()[-300:])
        except Exception:
            pass
        try:
            _finish_run(st, "Internal error")
        except Exception:
            pass
        return None
    if not cont:
        return None
    _redraw()
    return TICK


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------
def model_items(self, context):
    return _MODEL_ITEMS


EMBED_ITEMS = [
    ("none", "None (keyword only)", "No embedding model; use fast keyword matching"),
    ("nomic-embed-text", "nomic-embed-text (~274 MB)", "Small, fast, good general text embeddings"),
    ("embeddinggemma", "embeddinggemma (~620 MB)", "Google EmbeddingGemma"),
    ("mxbai-embed-large", "mxbai-embed-large (~670 MB)", "Higher quality, larger"),
    ("bge-m3", "bge-m3 (~1.2 GB)", "Multilingual, high quality"),
    ("all-minilm", "all-minilm (~46 MB)", "Tiny and fastest, lower quality"),
]


class AICtrlPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__
    use_memory: BoolProperty(
        name="Use memory (RAG)", default=True,
        description="Inject relevant past examples into the AI prompt")
    system_prompt: StringProperty(
        name="System prompt", default="",
        description="Extra standing instructions always added to the AI's system prompt "
                    "(units, style, naming, ...)")
    embed_model: EnumProperty(
        name="Embedding model", default="none", items=EMBED_ITEMS,
        description="Embedding model for semantic memory retrieval; None = keyword only")

    def draw(self, context):
        L = self.layout
        L.prop(self, "use_memory")
        L.label(text="Successful Agent runs are auto-saved to local memory.")
        L.separator()
        L.label(text="System prompt (always sent to the AI):")
        L.prop(self, "system_prompt", text="")
        L.separator()
        L.label(text="Semantic retrieval (optional)")
        row = L.row()
        row.prop(self, "embed_model", text="Model")
        row.label(text="In use: %s" % ("keyword" if self.embed_model == "none" else self.embed_model))
        dr = L.row()
        dr.enabled = (self.embed_model != "none") and (not _PULL["active"])
        dr.operator("ai.pull_embed", icon='IMPORT')
        if _PULL.get("status"):
            L.label(text="Download: %s" % _PULL["status"])
        L.label(text="Works without downloading — it falls back to keyword search.", icon='INFO')
        L.label(text="Download fetches an Ollama model, not any Blender file.", icon='INFO')
        L.label(text="Saved memories: %d" % count_episodes())


class ReportItem(bpy.types.PropertyGroup):
    text: StringProperty(default="")


class TodoItem(bpy.types.PropertyGroup):
    name: StringProperty(default="")
    status: StringProperty(default="pending")


class AIProps(bpy.types.PropertyGroup):
    user_input: StringProperty(name="Prompt", description="Describe what you want in natural language", default="")
    model: EnumProperty(name="Model", items=model_items, update=_on_model_change)
    mode: EnumProperty(
        name="Mode", default="AGENT",
        items=[
            ("ASK", "Ask", "No tools, only answers questions"),
            ("PLAN", "Plan", "No tools, builds a todo plan, does not execute"),
            ("AGENT", "Agent", "Unlocks all tools, actually executes"),
        ],
    )
    permission: EnumProperty(
        name="Permission", default="accept",
        description="Only effective in Agent mode",
        items=[
            ("read", "read", "Plan / read only, no modification"),
            ("accept", "accept", "Ask before every modifying step"),
            ("auto", "auto", "Run fully automatically"),
        ],
    )
    auto_retry: IntProperty(name="Auto-retry", default=2, min=0, max=5,
                            description="Times the AI auto-fixes and retries a failing step")
    safe_scan: BoolProperty(
        name="Safe scan", default=True,
        description="Scan AI-generated code and block dangerous calls (file/system/network) before running")
    step_vision: BoolProperty(
        name="Step vision check", default=False,
        description="Agent screenshots after every step for a vision model to review (slower; needs a vision model such as qwen3.6)")
    prefer_nodes: BoolProperty(
        name="Prefer procedural nodes", default=True,
        description="Nudge the AI to prefer Geometry Nodes + Shader nodes for procedural geometry and materials")
    use_polyhaven: BoolProperty(
        name="Use PolyHaven", default=True,
        description="Allow the AI to download free HDRIs and PBR materials from PolyHaven (needs network)")
    status: StringProperty(default="Ready (press Refresh to connect Ollama)")
    task_name: StringProperty(default="")
    connected: BoolProperty(default=False)
    loaded_info: StringProperty(default="none")
    caps: StringProperty(default="")
    caps_known: BoolProperty(default=False)
    model_vision: BoolProperty(default=False)
    is_running: BoolProperty(default=False)
    needs_decision: BoolProperty(default=False)
    needs_approve: BoolProperty(default=False)
    needs_vdecision: BoolProperty(default=False)
    decision: StringProperty(default="NONE")
    pending_name: StringProperty(default="")
    pending_code: StringProperty(default="")
    vissue: StringProperty(default="")
    report: CollectionProperty(type=ReportItem)
    report_index: IntProperty(default=0)
    todo: CollectionProperty(type=TodoItem)
    todo_index: IntProperty(default=0)


# ---------------------------------------------------------------------------
# UIList
# ---------------------------------------------------------------------------
class AI_UL_report(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        layout.label(text=item.text if item.text else " ")


class AI_UL_todo(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_prop, index):
        layout.label(text=item.name, icon=TODO_ICON.get(item.status, 'DOT'))


# ---------------------------------------------------------------------------
# Operators: connect / unload
# ---------------------------------------------------------------------------
class AI_OT_refresh(bpy.types.Operator):
    bl_idname = "ai.refresh"
    bl_label = "Refresh (connect / load models)"
    bl_description = "Check the Ollama connection and refresh the model list"

    def execute(self, context):
        global _MODEL_ITEMS
        st = context.scene.ai_props
        try:
            names = list_models()
        except Exception as e:
            st.connected = False
            _MODEL_ITEMS = [("__none__", "(cannot connect to Ollama)", "")]
            st.status = "Cannot connect to Ollama: %s" % e
            self.report({'ERROR'}, "Cannot connect to Ollama; make sure `ollama serve` is running")
            return {'CANCELLED'}
        st.connected = True
        if not names:
            _MODEL_ITEMS = [("__none__", "(no models downloaded)", "")]
            st.status = "Connected, but no models"
            return {'FINISHED'}
        _MODEL_ITEMS = [(n, n, "") for n in names]
        chosen = next((r for r in RECOMMENDED if r in names), names[0])
        try:
            st.model = chosen
        except Exception:
            pass
        try:
            refresh_caps(st)
        except Exception:
            pass
        st.loaded_info = loaded_text()
        st.status = "Connected, %d models loaded" % len(names)
        return {'FINISHED'}


class AI_OT_unload(bpy.types.Operator):
    bl_idname = "ai.unload"
    bl_label = "Unload (free memory)"
    bl_description = "Free the loaded model from memory to reclaim VRAM; the model stays on disk"

    def execute(self, context):
        st = context.scene.ai_props
        if st.is_running:
            self.report({'WARNING'}, "A task is running")
            return {'CANCELLED'}
        try:
            running = list_running()
        except Exception as e:
            self.report({'ERROR'}, "Cannot connect to Ollama: %s" % e)
            return {'CANCELLED'}
        if not running:
            st.loaded_info = "none"
            self.report({'INFO'}, "No model is currently loaded")
            return {'FINISHED'}
        failed = []
        for name, _ in running:
            try:
                unload_model(name)
            except Exception:
                failed.append(name)
        st.loaded_info = loaded_text()
        if failed:
            self.report({'WARNING'}, "Some failed to unload: %s" % ", ".join(failed))
        else:
            self.report({'INFO'}, "Unloaded; memory freed")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators: run / stop / beautify
# ---------------------------------------------------------------------------
class AI_OT_run(bpy.types.Operator):
    bl_idname = "ai.run"
    bl_label = "Run"
    bl_description = "Send to the AI according to the current mode"

    def execute(self, context):
        st = context.scene.ai_props
        if st.is_running:
            self.report({'WARNING'}, "The AI is running; wait or press Force stop")
            return {'CANCELLED'}
        if not st.user_input.strip():
            self.report({'WARNING'}, "Enter a prompt first")
            return {'CANCELLED'}
        if not st.model or st.model == "__none__":
            self.report({'WARNING'}, "Press Refresh and pick a model first")
            return {'CANCELLED'}

        snap = scene_snapshot() if st.mode == "AGENT" else ""
        prefs = _get_prefs()
        mem_ctx = ""
        extra_system = ""
        if prefs:
            extra_system = (prefs.system_prompt or "").strip()
            if prefs.use_memory and st.mode in ("PLAN", "AGENT"):
                mem_ctx = build_examples_context(st.user_input)
        R.reset(model=st.model, user_input=st.user_input, mode=st.mode,
                permission=st.permission, auto_retry=st.auto_retry, scene_snap=snap,
                step_vision=(st.step_vision and (st.model_vision or not st.caps_known)),
                prefer_nodes=st.prefer_nodes,
                use_polyhaven=st.use_polyhaven, safe_scan=st.safe_scan,
                memory_ctx=mem_ctx, extra_system=extra_system)
        st.is_running = True
        st.needs_decision = False
        st.needs_approve = False
        st.needs_vdecision = False
        st.decision = "NONE"
        st.task_name = ""
        st.todo.clear()
        report_add(st, "> [%s] %s" % (st.mode, st.user_input))

        if st.mode == "ASK":
            st.status = "Thinking..."
            R.phase = "ASK_WAIT"
            R.start_ask()
        elif st.mode == "PLAN" or (st.mode == "AGENT" and st.permission == "read"):
            st.status = "Planning..."
            R.phase = "PLANSHOW_WAIT"
            R.start_plan(include_scene=(st.mode == "AGENT"))
        else:
            st.status = "Planning..."
            R.phase = "PLAN_WAIT"
            R.start_plan(include_scene=True)

        if not bpy.app.timers.is_registered(_tick):
            bpy.app.timers.register(_tick, first_interval=0.1)
        _redraw()
        return {'FINISHED'}


class AI_OT_stop(bpy.types.Operator):
    bl_idname = "ai.stop"
    bl_label = "Force stop"
    bl_description = "Immediately interrupt the AI reply / execution"

    def execute(self, context):
        R.stop = True
        context.scene.ai_props.status = "Stopping..."
        return {'FINISHED'}


class AI_OT_beautify(bpy.types.Operator):
    bl_idname = "ai.beautify"
    bl_label = "Beautify scene"
    bl_description = "One click: EEVEE material preview + sky environment light + a sun + default PBR materials"

    def execute(self, context):
        st = context.scene.ai_props
        if st.is_running:
            self.report({'WARNING'}, "A task is running")
            return {'CANCELLED'}
        try:
            beautify_scene()
        except Exception as e:
            self.report({'ERROR'}, "Beautify failed: %s" % e)
            return {'CANCELLED'}
        self.report({'INFO'}, "Applied sky lighting + material preview")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators: decisions (approve/skip/cancel/visual-fix/visual-continue share one)
# ---------------------------------------------------------------------------
class AI_OT_decide(bpy.types.Operator):
    bl_idname = "ai.decide"
    bl_label = "Decide"
    value: StringProperty(default="NONE")

    def execute(self, context):
        context.scene.ai_props.decision = self.value
        return {'FINISHED'}


class AI_OT_mem_save(bpy.types.Operator):
    bl_idname = "ai.mem_save"
    bl_label = "Save last run"
    bl_description = "Save the last successful run to your local memory"

    def execute(self, context):
        if not _LAST_RUN.get("code"):
            self.report({'WARNING'}, "No successful run to save yet")
            return {'CANCELLED'}
        record_episode(_LAST_RUN["prompt"], _LAST_RUN["code"], _LAST_RUN.get("mode", "AGENT"), "saved")
        self.report({'INFO'}, "Saved to memory")
        return {'FINISHED'}


class AI_OT_mem_clear(bpy.types.Operator):
    bl_idname = "ai.mem_clear"
    bl_label = "Clear memory"
    bl_description = "Delete your saved memories (the shipped seed examples are kept)"

    def execute(self, context):
        try:
            p = _episodes_path()
            if os.path.exists(p):
                os.remove(p)
        except Exception as e:
            self.report({'ERROR'}, "%s" % e)
            return {'CANCELLED'}
        self.report({'INFO'}, "Memory cleared")
        return {'FINISHED'}


class AI_OT_pull_embed(bpy.types.Operator):
    bl_idname = "ai.pull_embed"
    bl_label = "Download embedding model"
    bl_description = "Download the embedding model via Ollama for semantic memory retrieval"

    def execute(self, context):
        prefs = _get_prefs()
        name = getattr(prefs, "embed_model", "none") if prefs else "none"
        if not name or name == "none":
            self.report({'WARNING'}, "Pick an embedding model first")
            return {'CANCELLED'}
        if _PULL["active"]:
            self.report({'WARNING'}, "A download is already running")
            return {'CANCELLED'}
        threading.Thread(target=_pull_worker, args=(name,), daemon=True).start()
        if not bpy.app.timers.is_registered(_pull_tick):
            bpy.app.timers.register(_pull_tick, first_interval=0.5)
        self.report({'INFO'}, "Downloading %s ..." % name)
        return {'FINISHED'}


class AI_OT_clear(bpy.types.Operator):
    bl_idname = "ai.clear"
    bl_label = "Clear"

    def execute(self, context):
        st = context.scene.ai_props
        if st.is_running:
            return {'CANCELLED'}
        st.user_input = ""
        st.task_name = ""
        st.report.clear()
        st.todo.clear()
        st.status = "Ready"
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
class AI_PT_panel(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "AI"
    bl_label = "AI Controller (Ollama)"

    def draw(self, context):
        st = context.scene.ai_props
        L = self.layout
        running = st.is_running

        rr = L.row()
        rr.enabled = not running
        rr.operator("ai.refresh", icon='FILE_REFRESH')
        L.label(text=("Ollama: connected" if st.connected else "Ollama: not connected"),
                icon=('CHECKMARK' if st.connected else 'X'))

        # Model
        box = L.box()
        box.label(text="Model", icon='OUTLINER_OB_LIGHT')
        mrow = box.row()
        mrow.enabled = not running
        mrow.prop(st, "model", text="")
        box.label(text="Loaded: %s" % st.loaded_info)
        box.label(text="Capabilities: %s" % (st.caps if st.caps else "(press Refresh)"))
        ur = box.row()
        ur.enabled = not running
        ur.operator("ai.unload", icon='UNLINKED')

        # Mode + permission + options (locked while running)
        mb = L.box()
        mb.enabled = not running
        mb.prop(st, "mode", text="Mode")
        prow = mb.row()
        prow.enabled = (st.mode == "AGENT")
        prow.prop(st, "permission", text="Permission")
        mb.prop(st, "auto_retry")
        mb.prop(st, "safe_scan")
        vr = mb.row()
        vr.enabled = (st.mode == "AGENT") and (st.model_vision or not st.caps_known)
        vr.prop(st, "step_vision")
        if st.caps_known and not st.model_vision:
            mb.label(text="vision check needs a vision model (e.g. qwen3.6)", icon='INFO')
        mb.prop(st, "prefer_nodes")
        mb.prop(st, "use_polyhaven")

        # Beautify
        bb = L.row()
        bb.enabled = not running
        bb.operator("ai.beautify", icon='SHADING_RENDERED')

        # Memory (RAG) — full settings live in Add-on Preferences
        prefs = _get_prefs()
        if prefs:
            memb = L.box()
            memb.label(text="Memory (RAG)", icon='PRESET')
            memb.label(text="On: %s    Saved: %d"
                       % ("yes" if prefs.use_memory else "no", count_episodes()))
            if _PULL.get("status"):
                memb.label(text="Embed download: %s" % _PULL["status"])
            memb.label(text="Auto-saves successful runs", icon='CHECKMARK')
            mrow = memb.row()
            mrow.enabled = not running
            mrow.operator("ai.mem_clear", icon='TRASH')
            memb.label(text="Settings in Add-on Preferences", icon='PREFERENCES')

        # Prompt (locked while running)
        col = L.column()
        col.enabled = not running
        col.label(text="Prompt:")
        col.prop(st, "user_input", text="")

        # Run / Force stop
        if running:
            sb = L.row()
            sb.scale_y = 1.4
            sb.alert = True
            sb.operator("ai.stop", icon='CANCEL', text="Force stop")
        else:
            run = L.row()
            run.scale_y = 1.3
            run.operator("ai.run", icon='PLAY')

        L.label(text=st.status)

        # Approval (accept)
        if st.needs_approve:
            ab = L.box()
            ab.label(text="Run this step?", icon='QUESTION')
            ab.label(text=st.pending_name)
            cb = ab.box()
            for ln in st.pending_code.split("\n"):
                for w in (textwrap.wrap(ln, WRAP) or [""]):
                    cb.label(text=w)
            r = ab.row(align=True)
            op = r.operator("ai.decide", text="Approve", icon='CHECKMARK')
            op.value = "APPROVE"
            op = r.operator("ai.decide", text="Skip", icon='TRIA_RIGHT')
            op.value = "SKIP"
            op = r.operator("ai.decide", text="Cancel", icon='X')
            op.value = "CANCEL"

        # Visual-issue decision
        if st.needs_vdecision:
            vb = L.box()
            vb.alert = True
            vb.label(text="AI thinks this step may have a problem:", icon='HIDE_OFF')
            for w in (textwrap.wrap(st.vissue, WRAP) or [""]):
                vb.label(text=w)
            r = vb.row(align=True)
            op = r.operator("ai.decide", text="Fix", icon='FILE_REFRESH')
            op.value = "VFIX"
            op = r.operator("ai.decide", text="Accept & continue", icon='CHECKMARK')
            op.value = "VCONT"

        # Failure decision
        if st.needs_decision:
            db = L.box()
            db.label(text="Step failed; please decide:", icon='ERROR')
            r = db.row(align=True)
            op = r.operator("ai.decide", text="Skip", icon='TRIA_RIGHT')
            op.value = "SKIP"
            op = r.operator("ai.decide", text="Cancel", icon='X')
            op.value = "CANCEL"

        # Todo list
        if len(st.todo) > 0:
            tb = L.box()
            tb.label(text="Todo (%s)" % st.task_name, icon='PRESET')
            tb.template_list("AI_UL_todo", "", st, "todo", st, "todo_index", rows=4)

        # AI report
        rb = L.box()
        rb.label(text="AI Report", icon='TEXT')
        rb.template_list("AI_UL_report", "", st, "report", st, "report_index", rows=8)

        cr = L.row()
        cr.enabled = not running
        cr.operator("ai.clear", icon='TRASH')


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------
CLASSES = (
    AICtrlPreferences,
    ReportItem,
    TodoItem,
    AIProps,
    AI_UL_report,
    AI_UL_todo,
    AI_OT_refresh,
    AI_OT_unload,
    AI_OT_run,
    AI_OT_stop,
    AI_OT_beautify,
    AI_OT_decide,
    AI_OT_mem_save,
    AI_OT_mem_clear,
    AI_OT_pull_embed,
    AI_OT_clear,
    AI_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ai_props = bpy.props.PointerProperty(type=AIProps)


def unregister():
    for _t in (_tick, _pull_tick):
        try:
            if bpy.app.timers.is_registered(_t):
                bpy.app.timers.unregister(_t)
        except Exception:
            pass
    if hasattr(bpy.types.Scene, "ai_props"):
        del bpy.types.Scene.ai_props
    for cls in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
