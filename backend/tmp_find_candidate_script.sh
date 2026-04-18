python3 - <<'PY'
import json, subprocess, requests
from collections import defaultdict

REF='gwvqlclkpquujxcbvvuj'
BASE=f'https://{REF}.supabase.co'

keys=json.loads(subprocess.check_output(['supabase','projects','api-keys','--project-ref',REF,'--output','json'], text=True))
service=next(k['api_key'] for k in keys if k.get('name')=='service_role')
H={'apikey':service,'Authorization':f'Bearer {service}'}

rows=requests.get(
    f"{BASE}/rest/v1/clothing_items?select=id,photo_id,category,description,colors,created_at&order=created_at.desc&limit=800",
    headers=H, timeout=30,
)
rows.raise_for_status()
items=rows.json()

by_photo=defaultdict(list)
for it in items:
    by_photo[it['photo_id']].append(it)

candidates=[]
for pid, arr in by_photo.items():
    txt=' '.join((it.get('description') or '').lower() for it in arr)
    # simple heuristic: black top + brown pants-ish in same photo
    has_black = 'black' in txt
    has_brown = 'brown' in txt or 'khaki' in txt or 'tan' in txt
    has_top = any(it.get('category')=='top' for it in arr)
    has_bottom = any(it.get('category')=='bottom' for it in arr)
    score = 0
    if has_black: score += 1
    if has_brown: score += 1
    if has_top: score += 1
    if has_bottom: score += 1
    if score >= 3:
        candidates.append((score, pid, arr))

candidates.sort(reverse=True)
print('candidate_count', len(candidates))
for score, pid, arr in candidates[:20]:
    descs=[(it['category'], it.get('description')) for it in arr[:8]]
    print('\nPID', pid, 'score', score, 'items', len(arr))
    for c,d in descs:
        print('-', c, '|', d)

# include storage path for top 10
pids=[pid for _,pid,_ in candidates[:10]]
if pids:
    q=','.join(pids)
    pr=requests.get(
      f"{BASE}/rest/v1/photos?select=id,job_id,storage_path,width,height,created_at&id=in.({q})",
      headers=H, timeout=30)
    pr.raise_for_status()
    print('\nSTORAGE_PATHS')
    for p in pr.json():
        print(p['id'], p['storage_path'])
PY