"""Organize illustration connections with KJNodes Set/Get routes."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def processing_group(node: dict, groups: list[dict], *, include_json: bool = False) -> int:
    x, y = node["pos"]
    candidates = [index for index, group in enumerate(groups)
                  if group["bounding"][2] >= 400 and (include_json or not group.get("title", "").startswith("JSON"))
                  and group["bounding"][0] <= x <= group["bounding"][0] + group["bounding"][2]
                  and group["bounding"][1] <= y <= group["bounding"][1] + group["bounding"][3]]
    if not candidates:
        raise ValueError(f'Node {node["id"]} has no processing group')
    return min(candidates, key=lambda index: groups[index]["bounding"][2] * groups[index]["bounding"][3])


def overlaps(nodes: list[dict], position: list[float], width: float = 220) -> bool:
    x, y = position
    for node in nodes:
        nx, ny = node["pos"]
        node_width, node_height = node["size"]
        if x < nx + node_width + 20 and x + width + 20 > nx and y - 40 < ny + node_height + 20 and y + 80 > ny - 30:
            return True
    return False


def route_position(nodes: list[dict], group: dict, *, width: int = 220,
                   preferred: list[float] | None = None) -> list[float]:
    gx, gy, group_width, height = group["bounding"]
    positions = [[gx + 30 + column * (width + 40), gy + 70 + row * 120]
                 for row in range(max(1, int((height - 130) / 120)))
                 for column in range(max(1, int((group_width - 40) / (width + 40))))]
    if preferred is not None:
        positions.sort(key=lambda point: (point[0] - preferred[0]) ** 2 + (point[1] - preferred[1]) ** 2)
        if gx + 20 <= preferred[0] and preferred[0] + width + 20 <= gx + group_width and gy + 60 <= preferred[1] <= gy + height - 80:
            positions.insert(0, preferred)
    position = next((point for point in positions if not overlaps(nodes, point, width)), None)
    if position is None:
        position = [gx + 30, gy + height + 60]
        while overlaps(nodes, position, width):
            position[1] += 120
        group["bounding"][3] = position[1] + 100 - gy
    return position


def organize_prompt_routes(workflow: dict) -> dict:
    result = copy.deepcopy(workflow)
    nodes, links = result["nodes"], result["links"]
    by_id = {node["id"]: node for node in nodes}
    consumers = [node for node in nodes if any(port.get("name") == "prompt_json" for port in node.get("inputs", []))]
    input_links = {port["link"] for node in consumers for port in node["inputs"] if port.get("name") == "prompt_json"}
    routes = [link for link in links if link[0] in input_links]
    sources = {(link[1], link[2]) for link in routes}
    if len(routes) != len(consumers) or len(sources) != 1:
        raise ValueError("Expected one shared JSON source connected to every prompt_json input")
    source_id, source_slot = next(iter(sources))
    prompt = by_id[source_id]
    if prompt["type"] != "PrimitiveStringMultiline":
        raise ValueError("The JSON prompt is already routed or has an unsupported source")
    route_name = "SOYA_PROMPT_JSON"
    if any(node["type"] == "SetNode" and node.get("widgets_values", [None])[0] == route_name for node in nodes):
        raise ValueError("The JSON route name is already in use")

    groups = result.get("groups", [])
    assignments: dict[int, list[dict]] = {}
    for node in consumers:
        index = processing_group(node, groups)
        assignments.setdefault(index, []).append(node)

    next_id = max(result.get("last_node_id", 0), *by_id) + 1
    next_link = max(result.get("last_link_id", 0), *(link[0] for link in links)) + 1
    prompt["size"] = [600, 460]
    setter = {
        "id": next_id, "type": "SetNode", "pos": [prompt["pos"][0] + 660, prompt["pos"][1] + 20],
        "size": [220, 60], "flags": {}, "order": len(nodes), "mode": 0,
        "inputs": [{"name": "STRING", "type": "STRING", "link": next_link}],
        "outputs": [{"name": "STRING", "type": "STRING", "links": [], "slot_index": 0}],
        "properties": {"previousName": route_name, "showOutputText": True},
        "widgets_values": [route_name], "title": "Set_" + route_name, "color": "#233", "bgcolor": "#355",
    }
    nodes.append(setter)
    for output in prompt.get("outputs", []):
        output["links"] = [identity for identity in output.get("links", []) if identity not in input_links]
    prompt["outputs"][source_slot]["links"].append(next_link)
    links.append([next_link, source_id, source_slot, next_id, 0, "STRING"])
    next_id += 1

    for index, members in sorted(assignments.items()):
        group = groups[index]
        position = route_position(nodes, group)
        member_ids = {node["id"] for node in members}
        local_links = [link for link in routes if link[3] in member_ids]
        getter = {
            "id": next_id, "type": "GetNode", "pos": position, "size": [220, 60],
            "flags": {}, "order": len(nodes), "mode": 0, "inputs": [],
            "outputs": [{"name": "STRING", "type": "STRING", "links": [link[0] for link in local_links], "slot_index": 0}],
            "properties": {"showOutputText": True}, "widgets_values": [route_name],
            "title": "Get_" + route_name, "color": "#233", "bgcolor": "#355",
        }
        nodes.append(getter)
        for link in local_links:
            link[1], link[2] = next_id, 0
        next_id += 1
    for group in groups:
        if group.get("title", "").startswith("JSON"):
            group["bounding"] = [prompt["pos"][0] - 40, prompt["pos"][1] - 80, 960, 600]
    result["last_node_id"] = next_id - 1
    result["last_link_id"] = next_link
    return result


def organize_model_routes(workflow: dict) -> dict:
    """Route MODEL, CLIP and VAE edges between processing groups independently."""
    result = copy.deepcopy(workflow)
    nodes, links, groups = result["nodes"], result["links"], result.get("groups", [])
    by_id = {node["id"]: node for node in nodes}
    group_ids = {node["id"]: processing_group(node, groups) for node in nodes
                 if any(link[5] in ("MODEL", "CLIP", "VAE") and node["id"] in (link[1], link[3])
                        for link in links) and node["type"] not in ("SetNode", "GetNode")}
    routes: dict[tuple[int, int, str], dict[int, list[list]]] = {}
    for link in links:
        _, source_id, source_slot, target_id, _, dtype = link
        if dtype not in ("MODEL", "CLIP", "VAE"):
            continue
        if by_id[source_id]["type"] in ("SetNode", "GetNode") or by_id[target_id]["type"] in ("SetNode", "GetNode"):
            continue
        if group_ids[source_id] != group_ids[target_id]:
            routes.setdefault((source_id, source_slot, dtype), {}).setdefault(group_ids[target_id], []).append(link)

    next_id = max(result.get("last_node_id", 0), *by_id) + 1
    next_link = max(result.get("last_link_id", 0), *(link[0] for link in links)) + 1
    names = {node["widgets_values"][0] for node in nodes
             if node["type"] == "SetNode" and node.get("widgets_values")}
    colors = {"MODEL": ("#223", "#335"), "CLIP": ("#432", "#653"), "VAE": ("#322", "#533")}

    for (source_id, source_slot, dtype), assignments in sorted(routes.items()):
        source = by_id[source_id]
        group = groups[group_ids[source_id]]
        title = group.get("title", "").upper()
        if "ILXL" in title or "XL" in title:
            family = "ILXL"
        elif "ANIMA" in title:
            family = "ANIMA"
        else:
            # Character-LoRA groups have no model family in their title.
            families = {"ILXL" if "XL" in groups[index].get("title", "").upper() else "ANIMA"
                        for index in assignments}
            if len(families) != 1:
                raise ValueError(f"Cannot determine model family for node {source_id}")
            family = families.pop()
        stage = ""
        if "FirstSampler" in source["type"]:
            stage = "_SAMPLED"
        elif "BatchLoraLoader" in source["type"]:
            stage = "_LORA" if "ANIMA" in title or "ILXL" in title else "_CHAR_LORA"
        elif dtype == "MODEL":
            stage = "_PATCHED" if "PATCH" in source.get("title", source["type"]).upper() else "_BASE"
        route_name = f"SOYA_{family}_{dtype}{stage}"
        if route_name in names:
            route_name += f"_{source_id}_{source_slot}"
        if route_name in names:
            raise ValueError(f"Route name already in use: {route_name}")
        names.add(route_name)
        color, bgcolor = colors[dtype]
        preferred = [source["pos"][0] + source["size"][0] + 50, source["pos"][1] + source_slot * 120]
        setter = {
            "id": next_id, "type": "SetNode", "pos": route_position(nodes, group, width=300, preferred=preferred),
            "size": [300, 60], "flags": {}, "order": len(nodes), "mode": 0,
            "inputs": [{"name": dtype, "type": dtype, "link": next_link}],
            "outputs": [{"name": dtype, "type": dtype, "links": [], "slot_index": 0}],
            "properties": {"previousName": route_name, "showOutputText": True},
            "widgets_values": [route_name], "title": "Set_" + route_name, "color": color, "bgcolor": bgcolor,
        }
        nodes.append(setter)
        routed_ids = {link[0] for local_links in assignments.values() for link in local_links}
        output = source["outputs"][source_slot]
        output["links"] = [identity for identity in (output.get("links") or []) if identity not in routed_ids]
        output["links"].append(next_link)
        links.append([next_link, source_id, source_slot, next_id, 0, dtype])
        next_id += 1
        next_link += 1
        for index, local_links in sorted(assignments.items()):
            getter = {
                "id": next_id, "type": "GetNode", "pos": route_position(nodes, groups[index], width=300),
                "size": [300, 60], "flags": {}, "order": len(nodes), "mode": 0, "inputs": [],
                "outputs": [{"name": dtype, "type": dtype, "links": [link[0] for link in local_links], "slot_index": 0}],
                "properties": {"showOutputText": True}, "widgets_values": [route_name],
                "title": "Get_" + route_name, "color": color, "bgcolor": bgcolor,
            }
            nodes.append(getter)
            for link in local_links:
                link[1], link[2] = next_id, 0
            next_id += 1
    result["last_node_id"] = next_id - 1
    result["last_link_id"] = next_link - 1
    return result


def _group_route_data(workflow: dict) -> dict:
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    virtual = {identity for identity, node in nodes.items() if node["type"] in ("SetNode", "GetNode")}
    groups = workflow["groups"]
    membership = {identity: groups[processing_group(node, groups, include_json=True)]["id"]
                  for identity, node in nodes.items() if identity not in virtual}

    from illustration_json import workflow_source_resolver
    resolve = workflow_source_resolver(workflow)

    direct = []
    for link in links.values():
        if link[3] not in virtual:
            item = list(link)
            item[1], item[2] = resolve(item[1], item[2])
            direct.append(item)
    set_nodes, get_nodes = {}, {}
    for node in nodes.values():
        if node["type"] == "SetNode":
            set_nodes[resolve(node["id"], 0)] = node
        elif node["type"] == "GetNode":
            source = resolve(node["id"], 0)
            for output in node.get("outputs", []):
                for link_id in output.get("links") or []:
                    target = links[link_id][3]
                    if target in membership:
                        get_nodes.setdefault((*source, membership[target]), node)
    return {"nodes": nodes, "virtual": virtual, "links": direct, "membership": membership,
            "sets": set_nodes, "gets": get_nodes, "groups": {group["id"]: group for group in groups}}


def organize_group_routes(workflow: dict, reference: dict, *, reference_groups=None, source_aliases=None) -> dict:
    """Keep IMAGE edges across groups and route other data through local Set/Get nodes."""
    result = copy.deepcopy(workflow)
    data = _group_route_data(result)
    template = _group_route_data(reference)
    reference_groups = reference_groups or {}
    source_aliases = source_aliases or {}
    nodes = [node for node in result["nodes"] if node["id"] not in data["virtual"]]
    links = data["links"]
    by_id = {node["id"]: node for node in nodes}
    routes = {}
    for link in links:
        if link[5] != "IMAGE" and data["membership"][link[1]] != data["membership"][link[3]]:
            routes.setdefault((link[1], link[2], link[5]), {}).setdefault(data["membership"][link[3]], []).append(link)
    next_id = max(result.get("last_node_id", 0), *(node["id"] for node in result["nodes"])) + 1
    next_link = max(result.get("last_link_id", 0), *(link[0] for link in result["links"])) + 1
    used_ids = set(by_id)
    used_names = set()
    colors = {"MODEL": ("#223", "#335"), "CLIP": ("#432", "#653"), "VAE": ("#322", "#533"), "STRING": ("#233", "#355")}

    def place(preferred, width, group):
        gx, gy, gw, gh = group["bounding"]
        x, y = preferred
        fits = gx <= x and x + width <= gx + gw and gy + 35 <= y and y + 60 <= gy + gh
        collision = any(x - 8 < n["pos"][0] + n["size"][0] and x + width + 8 > n["pos"][0]
                        and y - 38 < n["pos"][1] + n["size"][1] and y + 68 > n["pos"][1] - 30 for n in nodes)
        return preferred if fits and not collision else route_position(nodes, group, width=width, preferred=preferred)

    def identity(existing):
        nonlocal next_id
        if existing and existing["id"] not in used_ids:
            value = existing["id"]
        else:
            value = next_id
            next_id += 1
        used_ids.add(value)
        return value

    for (source_id, source_slot, dtype), assignments in sorted(routes.items()):
        source = by_id[source_id]
        alias = source_aliases.get((source_id, source_slot), (source_id, source_slot))
        old_set = data["sets"].get((source_id, source_slot))
        hint = template["sets"].get(alias)
        route_name = old_set["widgets_values"][0] if old_set else (hint["widgets_values"][0] if hint else f"SOYA_{dtype}_{source_id}_{source_slot}")
        if route_name in used_names:
            raise ValueError(f"Duplicate route name: {route_name}")
        used_names.add(route_name)
        color, bgcolor = colors.get(dtype, ("#333", "#555"))
        width = int((hint or old_set or {}).get("size", [300, 60])[0])
        preferred = [source["pos"][0] + source["size"][0] + 70, source["pos"][1]]
        if hint:
            reference_source = template["nodes"][alias[0]]
            preferred = [source["pos"][i] + hint["pos"][i] - reference_source["pos"][i] for i in (0, 1)]
        group = data["groups"][data["membership"][source_id]]
        setter = {"id": identity(old_set), "type": "SetNode", "pos": place(preferred, width, group),
                  "size": [width, 60], "flags": {}, "order": len(nodes), "mode": 0,
                  "inputs": [{"name": dtype, "type": dtype, "link": next_link}],
                  "outputs": [{"name": dtype, "type": dtype, "links": [], "slot_index": 0}],
                  "properties": {"previousName": route_name, "showOutputText": True}, "widgets_values": [route_name],
                  "title": "Set_" + route_name, "color": color, "bgcolor": bgcolor}
        nodes.append(setter)
        links.append([next_link, source_id, source_slot, setter["id"], 0, dtype])
        next_link += 1
        for group_id, local_links in sorted(assignments.items()):
            group = data["groups"][group_id]
            reference_group_id = reference_groups.get(group_id, group_id)
            hint = template["gets"].get((*alias, reference_group_id))
            old_get = data["gets"].get((source_id, source_slot, group_id))
            width = int((hint or old_get or {}).get("size", [300, 60])[0])
            preferred = [group["bounding"][0] + 30, group["bounding"][1] + 70]
            if hint:
                reference_group = template["groups"][reference_group_id]
                preferred = [group["bounding"][i] + hint["pos"][i] - reference_group["bounding"][i] for i in (0, 1)]
            getter = {"id": identity(old_get), "type": "GetNode", "pos": place(preferred, width, group),
                      "size": [width, 60], "flags": {}, "order": len(nodes), "mode": 0, "inputs": [],
                      "outputs": [{"name": dtype, "type": dtype, "links": [link[0] for link in local_links], "slot_index": 0}],
                      "properties": {"showOutputText": True}, "widgets_values": [route_name],
                      "title": "Get_" + route_name, "color": color, "bgcolor": bgcolor}
            nodes.append(getter)
            for link in local_links:
                link[1], link[2] = getter["id"], 0
    by_id = {node["id"]: node for node in nodes}
    for node in nodes:
        for output in node.get("outputs", []):
            output["links"] = []
    for link in links:
        by_id[link[1]]["outputs"][link[2]]["links"].append(link[0])
    result.update(nodes=nodes, links=links, last_node_id=max(used_ids), last_link_id=next_link - 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--model-routes", action="store_true", help="Organize MODEL, CLIP and VAE instead of the JSON prompt")
    mode.add_argument("--group-reference", type=Path,
                      help="Keep only IMAGE links between groups, using this workflow's Set/Get placement")
    args = parser.parse_args()
    workflow = json.loads(args.source.read_text(encoding="utf-8"))
    if args.group_reference:
        reference = json.loads(args.group_reference.read_text(encoding="utf-8"))
        result = organize_group_routes(workflow, reference)
    else:
        organize = organize_model_routes if args.model_routes else organize_prompt_routes
        result = organize(workflow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f'{args.output.name}: {sum(node["type"] == "SetNode" for node in result["nodes"])} SetNodes, '
          f'{sum(node["type"] == "GetNode" for node in result["nodes"])} GetNodes')


if __name__ == "__main__":
    main()
