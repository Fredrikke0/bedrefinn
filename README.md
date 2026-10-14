# BedreFINN

Requires mise, uv, and a running Podman or Docker engine. On macOS:

```sh
brew install mise uv podman
```

Start Podman with `podman machine start` (`podman machine init` once for a new
machine), or start Docker Desktop.

## Run

From the project directory:

```sh
mise trust
mise setup
mise run:download
mise run:site
```

Open http://127.0.0.1:8001. Setup preserves existing data.

## Model

Default: `google/siglip2-base-patch16-256`. For a faster, weaker model:

```sh
export BEDREFIN_MODEL_ID=google/siglip2-base-patch32-256
mise run:download --reembed-only
mise run:site
```

Stop ingestion and the site before switching models. Re-embedding preserves
listings and downloaded images. Use the same model in every terminal; finish
re-embedding before starting the site. `--clean-run` deletes all existing data
and local images.

