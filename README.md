# BedreFINN

Search FINN furniture and interior listings in the Oslo area by text or similar
images. Kitchenware, tableware, and decorations are excluded.

## Setup

Requires macOS 15 or newer, mise, uv, and a running Podman or Docker engine.
With Homebrew:

```sh
brew install mise uv podman
```

Start Podman with `podman machine start` (`podman machine init` once for a new
machine), or start Docker Desktop.

From the project directory:

```sh
mise trust
mise setup
```

Prepare the search models once; skip this if they have already been converted:

```sh
uv run python convert_coreml.py --batch-size 1
uv run python convert_coreml.py --batch-size 32
uv run python convert_coreml.py --text
```

## Run

Download listings; rerun this task whenever you want to refresh them:

```sh
mise run:download
```

Press Ctrl+C to stop and wait for pending images to finish processing. Then start
the site:

```sh
mise run:site
```

Open http://127.0.0.1:8001. Click a result image to find similar listings, and add
text to refine those matches.

Setup and normal downloads preserve existing data. `--clean-run` deletes all
listings and local images.
