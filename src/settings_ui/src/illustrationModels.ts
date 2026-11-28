import { useCallback, useEffect, useState } from "react";
import { usePageVisible } from "./ModalWorkspace";

type WorkflowItem = { id: string; name: string; filename: string; bindings: string[]; source: string; localPaths: string[] };
type WorkflowGroup = { category: string; name: string; itemIds: string[] };
type WorkflowCatalog = { items: WorkflowItem[]; defaultGroups: WorkflowGroup[]; workflowFolder: string; folderError: string };
type WorkflowGroups = { groups: WorkflowGroup[]; useDefaults: boolean };
export type IllustrationModel = { profile: string; label: string; workflows: { path: string; label: string }[]; uninstalledWorkflows: { id: string; label: string }[] };
const modelLabels: Record<string, string> = { moss: "MOSS", v3_anima: "ANIMA", v3: "ANIMA+ILXL" };
export const illustrationModelLabel = (profile: string) => modelLabels[profile] || ({ chansub: "챈섭", chansub_v3_anima: "챈섭 + ANIMA" } as Record<string, string>)[profile] || profile;

// MOSS runs on the ANIMA profile but is a distinct entry in the model picker.
export function isMossWorkflow(path: string): boolean {
  return /moss/i.test(path.replaceAll("\\", "/").split("/").pop() || "");
}

export function illustrationPresetFamily(preset: { config?: Record<string, unknown> } | null | undefined): "moss" | "anima" {
  const selected = String(preset?.config?.illustration_model_family || "").toLowerCase();
  if (selected === "moss" || selected === "anima") return selected;
  // Presets saved before the model selection existed fall back to the file name.
  const paths = Object.values((preset?.config?.illustration_workflow_source_paths || {}) as Record<string, unknown>);
  return paths.some(value => isMossWorkflow(String(value))) ? "moss" : "anima";
}

export function workflowPathKey(path: string): string {
  const normalized = path.replaceAll("\\", "/");
  return /^[a-z]:\//i.test(normalized) || normalized.startsWith("//") ? normalized.toLowerCase() : normalized;
}

// DUAL_VAE workflows belong to the emotion illustration workshop only.
export function isDualVaeWorkflow(path: string): boolean {
  return /dual[_\s-]?vae/i.test(path);
}

const pathLabel = (path: string) => `${path.replaceAll("\\", "/").split("/").pop()?.replace(/\.json$/i, "")} · 현재 설정`;
const addWorkflow = (model: IllustrationModel, path: string, label: string, unshift = false) => {
  if (model.workflows.some(workflow => workflowPathKey(workflow.path) === workflowPathKey(path))) return;
  const entry = { path, label: label.replace(/\.json$/i, "") };
  if (unshift) model.workflows.unshift(entry); else model.workflows.push(entry);
};

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
      addWorkflow(model, `${catalog.workflowFolder.replace(/[\\/]+$/, "")}/${relative}`,
        item.localPaths.length > 1 ? relative : item.name || item.filename);
    }
    models.set(profile, model);
  }
  // MOSS is a separate model choice. Pick up its workflows even when they are
  // registered outside a group named MOSS.
  if (labels.moss && !models.has("moss")) {
    const itemIds = new Set(groups.filter(group => matchesCategory(group.category)).flatMap(group => group.itemIds));
    const items = catalog.items.filter(item => itemIds.has(item.id) && isMossWorkflow(item.filename || item.name));
    const model: IllustrationModel = { profile: "moss", label: labels.moss, workflows: [], uninstalledWorkflows: [] };
    for (const item of items) {
      if (item.localPaths.length) continue;
      if (!model.uninstalledWorkflows.some(workflow => workflow.id === item.id)) {
        model.uninstalledWorkflows.push({ id: item.id, label: (item.name || item.filename).replace(/\.json$/i, "") });
      }
    }
    for (const item of items) for (const relative of item.localPaths) {
      addWorkflow(model, `${catalog.workflowFolder.replace(/[\\/]+$/, "")}/${relative}`,
        item.localPaths.length > 1 ? relative : item.name || item.filename);
    }
    if (model.workflows.length || model.uninstalledWorkflows.length) models.set("moss", model);
  }
  // Keep an existing selection usable when an edited workflow no longer matches its pack hash.
  for (const [profile, label] of Object.entries(labels)) {
    const path = configured[profile];
    if (!path) continue;
    const model = models.get(profile) || { profile, label, workflows: [], uninstalledWorkflows: [] };
    addWorkflow(model, path, pathLabel(path), true);
    models.set(profile, model);
  }
  // A MOSS preset stores its workflow on the ANIMA profile, so carry that
  // selection into the MOSS model entry.
  if (labels.moss) {
    const paths = Object.values(configured).filter(isMossWorkflow);
    const model = models.get("moss") || { profile: "moss", label: labels.moss, workflows: [], uninstalledWorkflows: [] };
    for (const path of paths) addWorkflow(model, path, pathLabel(path), true);
    if (model.workflows.length || model.uninstalledWorkflows.length) models.set("moss", model);
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
  const pageVisible = usePageVisible();
  useEffect(() => {
    if (!pageVisible) return;
    const onFocus = () => { if (!document.hidden) reload(); };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [pageVisible, reload]);
  return { models: data ? illustrationModels(data.catalog, data.groups, configured, category, labels, bindings) : [], loading, error: error || data?.catalog.folderError || "", reload };
}
