# mobile

Expo app for camera-roll ingestion, face selection, and closet labeling display.

## Flow

1. Request photo-library permission.
2. Read recent camera-roll photos and preprocess them (HEIC conversion / resize).
3. Upload photos to FastAPI backend.
4. Poll job status.
5. Show face clusters for user selection.
6. Show labeled clothing items while backend lookup promotes items from `pending` -> `exact` / `similar` / `generic`.

## Run

```bash
npm install
npx expo start
```

Set backend URL (recommended on physical device):

```bash
EXPO_PUBLIC_BACKEND_URL=http://<your-lan-ip>:8000
```

If unset:

- Android emulator defaults to `http://10.0.2.2:8000`
- iOS/Expo defaults to `http://<expo-host>:8000` when possible
- fallback is `http://localhost:8000`

## Notes

- Polling continues after `done` until no `pending` items remain.
- Item cards include tier badges and best-match link cards when available.
