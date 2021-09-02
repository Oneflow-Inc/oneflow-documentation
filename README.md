# Oneflow-Documentation

The source code of website https://docs.oneflow.org

Build the documentation locally:

```shell
python3 -m pip install -r requirements.txt
```

And then, change the directory to `cn/` or `en/`, run commands:

```shell
mkdocs build
```

The output HTML files will be generated at `site/` directory.

## Deployment

Run commands:

```shell
sh ./scripts/run-mike.sh
```

The multi-version website will be built and deployed at branch `docs_output`.
