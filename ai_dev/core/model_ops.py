from __future__ import annotations

import json


def resolve_model_for_tag(models: list[dict], tag: str, task_tag_aliases: dict[str, list[str]]) -> str:
    normalized = (tag or "").strip().lower()
    preferred_tags = task_tag_aliases.get(normalized, [normalized, "default"])

    for wanted in preferred_tags:
        for m in models:
            tags = [str(t).lower() for t in m.get("tags", [])]
            if wanted in tags:
                return m.get("name", "local-mlx")

    if models:
        return models[0].get("name", "local-mlx")
    return "local-mlx"


def command_route_model(args, load_config_fn, task_tag_aliases: dict[str, list[str]]) -> int:
    cfg = load_config_fn()
    models = cfg.get("models", [])
    chosen = resolve_model_for_tag(models, args.task_tag, task_tag_aliases)
    if args.json:
        print(json.dumps({"task_tag": args.task_tag, "model": chosen}, indent=2))
    else:
        print(chosen)
    return 0


def command_models(args, load_config_fn) -> int:
    cfg = load_config_fn()
    models = cfg.get("models", [])

    if args.json:
        print(json.dumps(models, indent=2))
        return 0

    if not models:
        print("No models configured in .ai-dev/config.json")
        return 0

    print("Configured model profiles:\n")
    for m in models:
        tags = ", ".join(m.get("tags", []))
        print(f"- {m.get('name', 'unnamed')}")
        print(f"  backend: {m.get('backend_model', '')}")
        print(f"  api_base: {m.get('api_base', '')}")
        if tags:
            print(f"  tags: {tags}")
    return 0


def command_configure_cursor(
    args,
    load_config_fn,
    write_file_fn,
    app_dir,
    task_tag_aliases: dict[str, list[str]],
) -> int:
    cfg = load_config_fn()

    selected_model = args.model
    if not selected_model and args.task_tag:
        selected_model = resolve_model_for_tag(cfg.get("models", []), args.task_tag, task_tag_aliases)

    if not selected_model:
        selected_model = cfg["cursor"]["model"]

    cursor_cfg = {
        "name": "Local LiteLLM",
        "provider": "openai",
        "baseUrl": args.base_url or cfg["cursor"]["base_url"],
        "apiKey": args.api_key or cfg["cursor"]["api_key"],
        "model": selected_model,
    }

    app_dir.mkdir(parents=True, exist_ok=True)
    output_path = app_dir / "cursor-openai.json"
    write_file_fn(output_path, json.dumps(cursor_cfg, indent=2) + "\n")

    print("Use the following OpenAI-compatible model config in Cursor:")
    print(json.dumps(cursor_cfg, indent=2))
    print(f"\nSaved: {output_path}")
    return 0
