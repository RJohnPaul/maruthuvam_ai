# Frontend (Vite + React)

This frontend is built with Vite and React, styled with Tailwind CSS.

## Quick start

From the `frontend` directory:

```bash
# 1) Install dependencies (uses a local cache to avoid EACCES errors)
npm install

# 2) Start the dev server
npm run dev
```

The dev server runs at:

- Local: <http://localhost:5173/>

> Tip: Use `npm run dev -- --host` to expose on your LAN.

## Why a local npm cache?

Some macOS setups hit npm EACCES permission errors when writing to the global cache. This project includes a `.npmrc` that sets a project-local cache at `./.npm-cache` so `npm install` works without sudo.

## Build for production

```bash
npm run build
```

Outputs to `dist/`.

## Lint

```bash
npm run lint
```

## Troubleshooting

- npm install shows EACCES errors: the local cache is already configured in `.npmrc`. If you still see issues, try clearing the cache: `rm -rf .npm-cache && npm install`.
- Dev server starts but page errors: check the terminal output for stack traces and ensure Node.js is v18+.
- Large bundle warning on build: consider code-splitting dynamic routes or using `manualChunks` in Vite if needed.
