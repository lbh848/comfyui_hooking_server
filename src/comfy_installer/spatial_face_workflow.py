"""Inspect project-owned spatial face context nodes in ComfyUI workflows."""

SPATIAL_MATCHER = "SoyaSpatialFaceContextV2_mdsoya"


def uses_spatial_faces(workflow):
    nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else workflow.values()
    return any(isinstance(n, dict) and n.get("type", n.get("class_type")) == SPATIAL_MATCHER for n in nodes)
