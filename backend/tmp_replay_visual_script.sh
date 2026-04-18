python3 - <<'PY'
import io
import json
import math
import subprocess
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

REF = 'gwvqlclkpquujxcbvvuj'
BASE = f'https://{REF}.supabase.co'
PHOTO_ID = 'e3e5f272-4887-48ca-8a86-ed13f7d2df53'

CATEGORY_QUERY_TERMS = {
    'top': ['shirt', 'top', 'sweater', 't-shirt', 'blouse'],
    'bottom': ['pants', 'trousers', 'shorts', 'skirt', 'jeans'],
    'dress': ['dress', 'gown'],
    'outerwear': ['jacket', 'coat', 'hoodie', 'blazer'],
    'shoes': ['shoes', 'sneakers', 'boots'],
    'hat': ['hat', 'cap', 'beanie'],
    'bag': ['bag', 'backpack', 'purse', 'handbag'],
    'accessory': ['watch', 'bracelet', 'necklace', 'tie', 'sunglasses', 'belt'],
}

# ---- load keys ----
env = {}
for line in Path('/Users/arjun/phiaHacksPersonalization/backend/.env').read_text().splitlines():
    s = line.strip()
    if not s or s.startswith('#') or '=' not in s:
        continue
    k, v = s.split('=', 1)
    env[k.strip()] = v.strip().strip('"').strip("'")

openai_key = env.get('OPENAI_API_KEY', '')
replicate_token = env.get('REPLICATE_API_TOKEN', '')
if not openai_key or not replicate_token:
    raise SystemExit('Missing OPENAI_API_KEY or REPLICATE_API_TOKEN')

keys = json.loads(subprocess.check_output([
    'supabase', 'projects', 'api-keys', '--project-ref', REF, '--output', 'json'
], text=True))
service = next(k['api_key'] for k in keys if k.get('name') == 'service_role')
sh = {'apikey': service, 'Authorization': f'Bearer {service}'}

# ---- fetch photo + items + face anchor ----
photo_res = requests.get(
    f"{BASE}/rest/v1/photos?select=id,job_id,storage_path,width,height&id=eq.{PHOTO_ID}&limit=1",
    headers=sh,
    timeout=30,
)
photo_res.raise_for_status()
photo = photo_res.json()[0]
job_id = photo['job_id']
image_url = f"{BASE}/storage/v1/object/public/photos/{photo['storage_path']}"

items_res = requests.get(
    f"{BASE}/rest/v1/clothing_items?select=id,category,description,bounding_box,created_at&photo_id=eq.{PHOTO_ID}&order=created_at.asc",
    headers=sh,
    timeout=30,
)
items_res.raise_for_status()
output_items = items_res.json()

sel_res = requests.get(
    f"{BASE}/rest/v1/selected_cluster?select=cluster_id&job_id=eq.{job_id}&limit=1",
    headers=sh,
    timeout=30,
)
sel_res.raise_for_status()
cluster_id = sel_res.json()[0]['cluster_id']

fd_res = requests.get(
    f"{BASE}/rest/v1/face_detections?select=bbox,confidence&photo_id=eq.{PHOTO_ID}&cluster_id=eq.{cluster_id}&order=created_at.asc&limit=1",
    headers=sh,
    timeout=30,
)
fd_res.raise_for_status()
face_bbox = fd_res.json()[0]['bbox']
face_tile_url = f"{BASE}/storage/v1/object/public/face-tiles/{job_id}/{PHOTO_ID}.jpg"

# ---- base image ----
img_res = requests.get(image_url, timeout=60)
img_res.raise_for_status()
base_img = Image.open(io.BytesIO(img_res.content)).convert('RGB')
W, H = base_img.size

# ---- run OpenAI extraction (person-focused) ----
outfit_schema = {
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "category": {"type": "string", "enum": ["top", "bottom", "dress", "outerwear", "shoes", "hat", "bag", "accessory"]},
          "description": {"type": "string"},
          "colors": {"type": "array", "items": {"type": "string"}},
          "pattern": {"type": "string"},
          "style": {"type": "string"},
          "brand_visible": {"type": ["string", "null"]},
          "bounding_box": {
            "type": "object",
            "properties": {"x": {"type": "number"}, "y": {"type": "number"}, "w": {"type": "number"}, "h": {"type": "number"}},
            "required": ["x", "y", "w", "h"],
            "additionalProperties": False,
          },
          "visibility": {"type": "string", "enum": ["clear", "partial", "obscured"]},
          "confidence": {"type": "number"},
        },
        "required": ["category", "description", "colors", "pattern", "style", "brand_visible", "bounding_box", "visibility", "confidence"],
        "additionalProperties": False,
      },
    }
  },
  "required": ["items"],
  "additionalProperties": False,
}

fbb = face_bbox
face_desc = f"face at normalized coordinates (x={fbb['left']:.3f}, y={fbb['top']:.3f}, width={fbb['width']:.3f}, height={fbb['height']:.3f})"
instructions = (
    "You are analyzing a photo to identify clothing items worn by a specific person. "
    "IMAGE 1 is a close-up crop of the target person's face. IMAGE 2 is the full photo. "
    f"Their {face_desc}; use that as a hint, but trust face matching from IMAGE 1. "
    "Identify every visible clothing or accessory item worn by this person only. "
    "Return short product-title-style descriptions."
)

openai_payload = {
    "model": "gpt-5.4",
    "reasoning": {"effort": "low"},
    "input": [{"role": "user", "content": [
        {"type": "input_text", "text": instructions},
        {"type": "input_image", "image_url": face_tile_url, "detail": "high"},
        {"type": "input_image", "image_url": image_url, "detail": "high"},
    ]}],
    "text": {
        "format": {
            "type": "json_schema",
            "name": "outfit_analysis",
            "schema": outfit_schema,
            "strict": True,
        }
    },
}

oraw = requests.post(
    'https://api.openai.com/v1/responses',
    headers={'Authorization': f'Bearer {openai_key}', 'Content-Type': 'application/json'},
    json=openai_payload,
    timeout=120,
)
oraw.raise_for_status()
odata = oraw.json()
otext = odata.get('output_text')
if not otext:
    for out in odata.get('output', []):
        for c in out.get('content', []):
            if c.get('type') == 'output_text' and c.get('text'):
                otext = c['text']
                break
        if otext:
            break
if not otext:
    raise SystemExit('No OpenAI output_text')

openai_items = (json.loads(otext).get('items') or [])

# ---- run Replicate detections ----
model_res = requests.get('https://api.replicate.com/v1/models/adirik/grounding-dino', headers={'Authorization': f'Bearer {replicate_token}'}, timeout=30)
model_res.raise_for_status()
version = (model_res.json().get('latest_version') or {}).get('id')
if not version:
    raise SystemExit('No latest model version')

terms = sorted({t for i in openai_items for t in CATEGORY_QUERY_TERMS.get(i.get('category', ''), [])})
if not terms:
    terms = ['shirt', 'pants', 'jacket', 'shoes']
query = ', '.join(terms)

pred = requests.post(
    'https://api.replicate.com/v1/predictions',
    headers={'Authorization': f'Bearer {replicate_token}', 'Content-Type': 'application/json', 'Prefer': 'wait=10'},
    json={
        'version': version,
        'input': {
            'image': image_url,
            'query': query,
            'box_threshold': 0.25,
            'text_threshold': 0.2,
            'show_visualisation': False,
        },
    },
    timeout=60,
)
pred.raise_for_status()
pdata = pred.json()
status = pdata.get('status')
get_url = (pdata.get('urls') or {}).get('get')
for _ in range(70):
    if status in ('succeeded', 'failed', 'canceled'):
        break
    if not get_url:
        break
    time.sleep(1)
    poll = requests.get(get_url, headers={'Authorization': f'Bearer {replicate_token}'}, timeout=30)
    poll.raise_for_status()
    pdata = poll.json(); status = pdata.get('status')
rep_dets = ((pdata.get('output') or {}).get('detections') or []) if status == 'succeeded' else []

# ---- draw helpers ----
try:
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 16)
except Exception:
    font = ImageFont.load_default(); small = ImageFont.load_default()

palette = [
    (239,68,68), (245,158,11), (34,197,94), (59,130,246), (168,85,247), (236,72,153), (20,184,166), (251,191,36),
    (244,63,94), (22,163,74), (14,165,233), (217,70,239),
]

def draw_label(draw, x1, y1, label, color, use_font):
    tb = draw.textbbox((0, 0), label, font=use_font)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    tx = x1
    ty = max(0, y1 - (th + 10))
    draw.rectangle((tx, ty, tx + tw + 12, ty + th + 8), fill=color)
    draw.text((tx + 6, ty + 4), label, fill=(255,255,255), font=use_font)

# OpenAI overlay
openai_overlay = base_img.copy()
do = ImageDraw.Draw(openai_overlay)
for i, it in enumerate(openai_items, start=1):
    bb = (it.get('bounding_box') or {})
    x1 = max(0, min(W-1, int(float(bb.get('x', 0))*W)))
    y1 = max(0, min(H-1, int(float(bb.get('y', 0))*H)))
    x2 = max(x1+1, min(W, int((float(bb.get('x', 0))+float(bb.get('w', 0)))*W)))
    y2 = max(y1+1, min(H, int((float(bb.get('y', 0))+float(bb.get('h', 0)))*H)))
    color = palette[(i-1) % len(palette)]
    do.rectangle((x1,y1,x2,y2), outline=color, width=5)
    draw_label(do, x1, y1, f"O{i}:{it.get('category','?')}", color, small)

# Replicate overlay
rep_overlay = base_img.copy()
dr = ImageDraw.Draw(rep_overlay)
for i, det in enumerate(rep_dets, start=1):
    b = det.get('bbox') or []
    if len(b) != 4:
        continue
    x1,y1,x2,y2 = [int(float(v)) for v in b]
    color = palette[(i-1) % len(palette)]
    dr.rectangle((x1,y1,x2,y2), outline=color, width=5)
    lbl = f"R{i}:{det.get('label','')} {float(det.get('confidence') or 0):.2f}"
    draw_label(dr, x1, y1, lbl, color, small)

# Output overlay (stored boxes)
out_overlay = base_img.copy()
du = ImageDraw.Draw(out_overlay)
for i, it in enumerate(output_items, start=1):
    bb = it.get('bounding_box') or {}
    x1 = max(0, min(W-1, int(float(bb.get('x', 0))*W)))
    y1 = max(0, min(H-1, int(float(bb.get('y', 0))*H)))
    x2 = max(x1+1, min(W, int((float(bb.get('x', 0))+float(bb.get('w', 0)))*W)))
    y2 = max(y1+1, min(H, int((float(bb.get('y', 0))+float(bb.get('h', 0)))*H)))
    color = palette[(i-1) % len(palette)]
    du.rectangle((x1,y1,x2,y2), outline=color, width=5)
    draw_label(du, x1, y1, f"F{i}:{it.get('category','?')}", color, small)

# comparison strip
pad = 20
title_h = 54
cmp_w = W*3 + pad*4
cmp_h = H + pad*2 + title_h
cmp = Image.new('RGB', (cmp_w, cmp_h), (248,250,252))
cd = ImageDraw.Draw(cmp)

panes = [
    (openai_overlay, 'OpenAI Produced Regions (person-targeted)'),
    (rep_overlay, f'Replicate Detections (query terms, {len(rep_dets)} boxes)'),
    (out_overlay, f'Final Output Regions Stored In Backend ({len(output_items)} boxes)'),
]

x = pad
for img, title in panes:
    cmp.paste(img, (x, pad + title_h))
    cd.rectangle((x, pad + title_h, x + W, pad + title_h + H), outline=(148,163,184), width=2)
    cd.text((x, pad + 16), title, fill=(15,23,42), font=small)
    x += W + pad

out_dir = Path('/Users/arjun/phiaHacksPersonalization/backend/data/visual-debug')
out_dir.mkdir(parents=True, exist_ok=True)
openai_path = out_dir / f'{PHOTO_ID}_openai_regions.jpg'
rep_path = out_dir / f'{PHOTO_ID}_replicate_regions.jpg'
final_path = out_dir / f'{PHOTO_ID}_final_regions.jpg'
cmp_path = out_dir / f'{PHOTO_ID}_comparison.jpg'

openai_overlay.save(openai_path, format='JPEG', quality=92)
rep_overlay.save(rep_path, format='JPEG', quality=92)
out_overlay.save(final_path, format='JPEG', quality=92)
cmp.save(cmp_path, format='JPEG', quality=92)

print('PHOTO_ID', PHOTO_ID)
print('OPENAI', openai_path)
print('REPLICATE', rep_path)
print('FINAL', final_path)
print('COMPARE', cmp_path)
print('QUERY', query)
print('OPENAI_ITEMS', len(openai_items))
print('REPLICATE_DETS', len(rep_dets))
print('FINAL_ITEMS', len(output_items))
PY