import { useCallback, useEffect, useState } from "react";

type WorkflowItem = { id: string; name: string; filename: string; bindings: string[]; source: string; localPaths: string[] };
type WorkflowGroup = { category: string; name: string; itemIds: string[] };
type WorkflowCatalog = { items: WorkflowItem[]; defaultGroups: WorkflowGroup[]; workflowFolder: string; folderError: string };
type WorkflowGroups = { groups: WorkflowGroup[]; useDefaults: boolean };
export type IllustrationModel = { profile: string; label: string; workflows: { path: string; label: string }[]; uninstalledWorkflows: { id: string; label: string }[] };
const modelLabels: Record<string, string> = { v3_anima: "ANIMA", v3: "ANIMA+ILXL" };
export const illustrationModelLabel = (profile: string) => modelLabels[profile] || ({ chansub: "챈섭", chansub_v3_anima: "챈섭 + ANIMA" } as Record<string, string>)[profile] || profile;

export function workflowPathKey(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  return /^[a-z]:\//i.test(normalized) || normalized.startsWith("//") ? normalized.toLowerCase() : normalized;
}

// DUAL_VAE workflows belong to the emotion illustration workshop only.
export function isDualVaeWorkflow(path: string): boolean {
  return /dual[_\s-]?vae/i.test(path);
}

export function illustrationModels(catalog: WorkflowCatalog, saved: WorkflowGroups, configured: Record<string, string>, category: string | readonly string[] = "삽화", labels = modelLabels, bindings: Record<string, string> = {}): IllustrationModel[] {
  const groups = saved.useDefaults ? catalog.defaultGroups : saved.groups;
  const models = new Map<string, IllustrationModel>();
  const matchesCategory = (name: string) => typeof category === "string" ? name === category : category.includes(name);
  for (const group of groups.filter(item => matchesCategory(item.category))) {
    const items = catalog.items.filter(item => {
      const memberships = item.source === "pack" ? catalog.defaultGroups : groups;
      return memberships.some(member => matchesCategory(member.category) && member.name === group.name && member.itemIds.includes(item.id));
    });
    // The engine supports these explicit profiles; model groups supply their labels and workflows.
    const profiles = Object.keys(labels).filter(profile => items.some(item => item.bindings.includes(bindings[profile] || `illustration_workflow_source_paths.${profile}`)));
    const profile = Object.entries(labels).find(([, label]) => label === group.name.replaceAll(" ", "").toUpperCase())?.[0]
      || (profiles.length === 1 ? profiles[0] : "");
    if (!profile) continue;
    const model: IllustrationModel = models.get(profile) || { profile, label: group.name, workflows: [], uninstalledWorkflows: [] };
    for (const item of items.filter(item => !item.localPaths.length)) {
      if (!model.uninstalledWorkflows.some(workflow => workflow.id === item.id)) {
        model.uninstalledWorkflows.push({ id: item.id, label: (item.name || item.filename).replace(/\.json$/i, "") });
      }
    }
    for (const item of items) for (const relative of item.localPaths) {
      const path = `${catalog.workflowFolder.replace(/[\\/]+$/, "")}/${relative}`;
      if (!model.workflows.some(workflow => workflowPathKey(workflow.path) === workflowPathKey(path))) {
        model.workflows.push({ path, label: (item.localPaths.length > 1 ? relative : item.name || item.filename).replace(/\.json$/i, "") });
      }
    }
    models.set(profile, model);
  }
  // Keep an existing selection usable when an edited workflow no longer matches its pack hash.
  for (const [profile, label] of Object.entries(labels)) {
    const path = configured[profile];
    if (!path) continue;
    const model = models.get(profile) || { profile, label, workflows: [], uninstalledWorkflows: [] };
    if (!model.workflows.some(workflow => workflowPathKey(workflow.path) === workflowPathKey(path))) {
      model.workflows.unshift({ path, label: `${path.replaceAll("\\", "/").split("/").pop()?.replace(/\.json$/i, "")} · 현재 설정` });
    }
    models.set(profile, model);
  }
  return [...models.values()];
}

export function useIllustrationModels(configured: Record<string, string>, category: string | readonly string[] = "삽화", labels = modelLabels, bindings: Record<string, string> = {}) {
  const [data, setData] = useState<{ catalog: WorkflowCatalog; groups: WorkflowGroups } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const reload = useCallback(() => setRevision(value => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError("");
    async function read(path: string) {
      const response = await fetch(`/api/setting/workflows${path}`, {
        headers: { "X-Comfy-Settings": "1", "X-Comfy-Settings-Schema": "2" }, cache: "no-store", signal: controller.signal,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result?.error || "그림 모델 목록을 불러오지 못했습니다.");
      return result;
    }
    void Promise.all([read(""), read("/groups")]).then(([catalog, groups]) => {
      if (!controller.signal.aborted) setData({ catalog, groups });
    }).catch(cause => { if (!controller.signal.aborted) setError(cause.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [revision]);
  useEffect(() => {
    window.addEventListener("focus", reload);
    return () => window.removeEventListener("focus", reload);
  }, [reload]);
  return { models: data ? illustrationModels(data.catalog, data.groups, configured, category, labels, bindings) : [], loading, error: error || data?.catalog.folderError || "", reload };
}
