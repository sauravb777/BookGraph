import type { GraphEdge, GraphNode } from "@/lib/graph-types";

export type GraphSearchResult = { nodes: GraphNode[]; edges: GraphEdge[] };
type ApiError = { detail?: unknown };

export async function searchBookGraph(query: string): Promise<GraphSearchResult> {
  const apiBase = process.env.NEXT_PUBLIC_API_URL?.trim();
  if (!apiBase) throw new Error("The BookGraph service is not connected yet.");

  let response: Response;
  try {
    response = await fetch(`${apiBase.endsWith("/") ? apiBase.slice(0, -1) : apiBase}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: 40 }),
      cache: "no-store",
    });
  } catch {
    throw new Error("BookGraph could not be reached. Please try again.");
  }

  const data = (await response.json().catch(() => ({}))) as ApiError & Partial<GraphSearchResult>;
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : "No meaningful book connections found. Try another word.";
    throw new Error(detail);
  }
  if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) {
    throw new Error("The graph could not be loaded. Please try again.");
  }
  return { nodes: data.nodes as GraphNode[], edges: data.edges as GraphEdge[] };
}
