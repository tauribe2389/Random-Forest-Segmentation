from __future__ import annotations

from typing import Any

DEFAULT_SCHEMA_VERSION = 1
DEFAULT_CLASS_NAMES = ["class_1", "class_2", "class_3"]


def parse_class_names(raw_value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(raw_value, str):
        for line in raw_value.replace(",", "\n").splitlines():
            item = str(line).strip()
            if item:
                tokens.append(item)
    elif isinstance(raw_value, list):
        for item in raw_value:
            if isinstance(item, dict):
                candidate = str(item.get("name", "")).strip()
            else:
                candidate = str(item).strip()
            if candidate:
                tokens.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in tokens:
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(item)
    return deduped


def build_schema_from_names(names: list[str], *, start_id: int = 1) -> dict[str, Any]:
    normalized = parse_class_names(names)
    next_id = max(1, int(start_id))
    classes: list[dict[str, Any]] = []
    for order, name in enumerate(normalized):
        classes.append({"id": next_id, "name": name, "order": order})
        next_id += 1
    return {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "next_class_id": next_id,
        "classes": classes,
    }


def normalize_class_schema(
    payload: Any,
    *,
    fallback_names: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("classes"), list):
        class_rows: list[dict[str, Any]] = []
        used_ids: set[int] = set()
        for index, row in enumerate(payload.get("classes", [])):
            if not isinstance(row, dict):
                continue
            try:
                class_id = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            if class_id <= 0 or class_id in used_ids:
                continue
            class_name = str(row.get("name", "")).strip()
            if not class_name:
                continue
            try:
                order = int(row.get("order", index))
            except (TypeError, ValueError):
                order = index
            used_ids.add(class_id)
            class_rows.append({"id": class_id, "name": class_name, "order": order})

        class_rows.sort(key=lambda item: (int(item["order"]), int(item["id"])))
        for order, row in enumerate(class_rows):
            row["order"] = order

        if not class_rows:
            return build_schema_from_names(fallback_names or DEFAULT_CLASS_NAMES)

        max_id = max(int(row["id"]) for row in class_rows)
        try:
            candidate_next = int(payload.get("next_class_id", max_id + 1))
        except (TypeError, ValueError):
            candidate_next = max_id + 1
        next_class_id = max(max_id + 1, candidate_next)
        return {
            "schema_version": DEFAULT_SCHEMA_VERSION,
            "next_class_id": next_class_id,
            "classes": class_rows,
        }

    if isinstance(payload, list):
        return build_schema_from_names(parse_class_names(payload))

    return build_schema_from_names(fallback_names or DEFAULT_CLASS_NAMES)


def class_entries(payload: Any, *, fallback_names: list[str] | None = None) -> list[dict[str, Any]]:
    schema = normalize_class_schema(payload, fallback_names=fallback_names)
    return [dict(item) for item in schema.get("classes", [])]


def class_names(payload: Any, *, fallback_names: list[str] | None = None) -> list[str]:
    return [str(item["name"]) for item in class_entries(payload, fallback_names=fallback_names)]


def class_name_by_id(payload: Any, *, fallback_names: list[str] | None = None) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for row in class_entries(payload, fallback_names=fallback_names):
        mapping[int(row["id"])] = str(row["name"])
    return mapping


def resolve_class_id(raw_value: Any, payload: Any, *, fallback_names: list[str] | None = None) -> int | None:
    try:
        class_id = int(raw_value)
    except (TypeError, ValueError):
        class_id = None

    entries = class_entries(payload, fallback_names=fallback_names)
    if class_id is not None and any(int(item["id"]) == class_id for item in entries):
        return class_id

    candidate_name = str(raw_value or "").strip().lower()
    if not candidate_name:
        return None
    for item in entries:
        if str(item["name"]).strip().lower() == candidate_name:
            return int(item["id"])
    return None


def update_schema_with_names(existing_payload: Any, proposed_names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    existing = class_entries(existing_payload)
    existing_by_id = {int(item["id"]): str(item["name"]) for item in existing}
    existing_ids = [int(item["id"]) for item in existing]

    proposed = parse_class_names(proposed_names)
    normalized_existing_by_name: dict[str, int] = {}
    for item in existing:
        normalized_existing_by_name[str(item["name"]).strip().lower()] = int(item["id"])

    planned: list[dict[str, Any]] = []
    used_ids: set[int] = set()
    for name in proposed:
        lowered = name.lower()
        class_id = normalized_existing_by_name.get(lowered)
        if class_id is not None and class_id not in used_ids:
            planned.append({"id": class_id, "name": name})
            used_ids.add(class_id)
        else:
            planned.append({"id": None, "name": name})

    unmatched_existing = [int(item["id"]) for item in existing if int(item["id"]) not in used_ids]
    try:
        next_class_id = int(normalize_class_schema(existing_payload).get("next_class_id", 1))
    except (TypeError, ValueError):
        next_class_id = 1
    next_class_id = max(next_class_id, (max(existing_ids) + 1) if existing_ids else 1)

    renamed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    for item in planned:
        class_id = item["id"]
        name = str(item["name"])
        if class_id is None:
            if unmatched_existing:
                class_id = unmatched_existing.pop(0)
                item["id"] = class_id
                before_name = existing_by_id.get(class_id, "")
                if before_name != name:
                    renamed.append({"id": class_id, "from": before_name, "to": name})
            else:
                class_id = next_class_id
                item["id"] = class_id
                next_class_id += 1
                added.append({"id": class_id, "name": name})
        else:
            before_name = existing_by_id.get(class_id, "")
            if before_name != name:
                renamed.append({"id": class_id, "from": before_name, "to": name})

    used_ids = {int(item["id"]) for item in planned}
    removed = [item for item in existing if int(item["id"]) not in used_ids]
    removed_ids = [int(item["id"]) for item in removed]
    removed_names = [str(item["name"]) for item in removed]

    surviving_existing_order = [class_id for class_id in existing_ids if class_id not in removed_ids]
    proposed_existing_order = [int(item["id"]) for item in planned if int(item["id"]) in existing_ids]
    reordered = surviving_existing_order != proposed_existing_order

    classes: list[dict[str, Any]] = []
    for order, row in enumerate(planned):
        classes.append(
            {
                "id": int(row["id"]),
                "name": str(row["name"]),
                "order": order,
            }
        )
    schema = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "next_class_id": max(next_class_id, (max((int(item["id"]) for item in classes), default=0) + 1)),
        "classes": classes,
    }
    diff = {
        "existing": [str(item["name"]) for item in existing],
        "proposed": [str(item["name"]) for item in classes],
        "added": [str(item["name"]) for item in added],
        "added_ids": [int(item["id"]) for item in added],
        "removed": removed_names,
        "removed_ids": removed_ids,
        "renamed": renamed,
        "reordered": reordered,
        "append_only": not removed_ids and not renamed and not reordered and len(classes) >= len(existing),
        "destructive": bool(removed_ids),
    }
    return schema, diff
