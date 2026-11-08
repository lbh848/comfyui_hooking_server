import type { IllustrationData } from "./illustration";

export interface FlowNode extends IllustrationData {
  id: string;
  label: string;
  status: string;
  kind?: string;
  executor?: string;
  dependencies?: string[];
  started_at?: number | null;
  ended_at?: number | null;
}
export interface IllustrationFlow {
  id: string;
  label: string;
  status: string;
  revision: number;
  created_at: number;
  updated_at: number;
  nodes: FlowNode[];
}
export function layoutNodes(nodes: FlowNode[], kind: string): {
  positions: Map<string, { depth: number; x: number; y: number }>;
  width: number;
  height: number;
};
export function nodePreviewKind(node: FlowNode): "mask" | "image" | "";
export function illustrationSlot(node: FlowNode): number | null;
export function imageMaskNode(node: FlowNode, flow: IllustrationFlow): FlowNode | undefined;
