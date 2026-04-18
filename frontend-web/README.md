# frontend-web

Next.js observer dashboard for backend pipeline progress.

## What it shows

- Job list + selected job detail.
- Live status polling (`uploading`, `analyzing_faces`, `awaiting_face_pick`, `extracting_clothing`, `done`, `failed`).
- Face cluster previews and selected cluster.
- Per-photo clothing items with tiers (`pending`, `exact`, `similar`, `generic`).
- Best-match cards (title/source/price/confidence/link).
- Per-item refresh button to retrigger backend lookup.

## Run

```bash
npm install
npm run dev
```

Set backend URL when needed:

```bash
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

Open `http://localhost:3000`.
