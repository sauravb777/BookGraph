export type GraphNode = {
  id: string;
  label: string;
  kind: "book" | "author" | "genre" | "theme" | "publisher" | "year";
  x: number;
  y: number;
  r: number;
  lines: string[];
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  relation: string;
  bend?: number;
  showLabel?: boolean;
};
