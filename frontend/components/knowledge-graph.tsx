"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent, PointerEvent, WheelEvent } from "react";
import type { GraphEdge, GraphNode } from "@/lib/graph-types";

type Props = { nodes: GraphNode[]; edges: GraphEdge[] };
type Transform = { x: number; y: number; scale: number };
type DragState = { startX: number; startY: number; baseX: number; baseY: number; captured: boolean };
type ViewBox = { x: number; y: number; width: number; height: number };

const fillByKind: Record<GraphNode["kind"], string> = {
  book: "#F4BFC2",
  author: "#91DCE8",
  genre: "#F7E18E",
  theme: "#B5E7B4",
  publisher: "#C8B9EB",
  year: "#E9D7CE",
};

function sketchCirclePath(cx: number, cy: number, radius: number, seed: number) {
  const points = 72;
  const coords: Array<[number, number]> = [];
  for (let index = 0; index < points; index += 1) {
    const angle = (Math.PI * 2 * index) / points;
    const wobble = 1 + 0.010 * Math.sin(angle * 3 + seed) + 0.006 * Math.cos(angle * 5 + seed * 0.73);
    coords.push([cx + Math.cos(angle) * radius * wobble, cy + Math.sin(angle) * radius * wobble]);
  }
  return coords.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`).join(" ") + " Z";
}

function edgeGeometry(source: GraphNode, target: GraphNode, bend = 0) {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const distance = Math.max(1, Math.hypot(dx, dy));
  const ux = dx / distance;
  const uy = dy / distance;
  const startX = source.x + ux * (source.r - 1);
  const startY = source.y + uy * (source.r - 1);
  const endX = target.x - ux * (target.r + 2);
  const endY = target.y - uy * (target.r + 2);
  const px = -uy;
  const py = ux;
  const controlX = (startX + endX) / 2 + px * bend;
  const controlY = (startY + endY) / 2 + py * bend;
  const midX = 0.25 * startX + 0.5 * controlX + 0.25 * endX;
  const midY = 0.25 * startY + 0.5 * controlY + 0.25 * endY;
  const rawAngle = (Math.atan2(endY - startY, endX - startX) * 180) / Math.PI;
  const normalizedAngle = rawAngle > 90 || rawAngle < -90 ? rawAngle + 180 : rawAngle;
  const angle = Math.max(-35, Math.min(35, normalizedAngle));
  return {
    path: `M ${startX.toFixed(2)} ${startY.toFixed(2)} Q ${controlX.toFixed(2)} ${controlY.toFixed(2)} ${endX.toFixed(2)} ${endY.toFixed(2)}`,
    midX,
    midY,
    angle,
  };
}

function labelFontSize(node: GraphNode) {
  const longestLine = Math.max(1, ...node.lines.map((line) => line.length));
  if (node.kind === "book") return longestLine >= 11 ? 22 : 25;
  return longestLine >= 12 ? 20 : 23;
}

function calculateViewBox(nodes: GraphNode[]): ViewBox {
  if (nodes.length === 0) return { x: 0, y: 0, width: 1800, height: 1100 };
  // Keep the canvas tightly framed so the larger network uses the available screen.
  const padding = nodes.length > 12 ? 34 : 55;
  const minX = Math.min(...nodes.map((node) => node.x - node.r)) - padding;
  const maxX = Math.max(...nodes.map((node) => node.x + node.r)) + padding;
  const minY = Math.min(...nodes.map((node) => node.y - node.r)) - padding;
  const maxY = Math.max(...nodes.map((node) => node.y + node.r)) + padding;
  let width = Math.max(1100, maxX - minX);
  let height = Math.max(760, maxY - minY);
  const targetAspect = 1.68;
  if (width / height < targetAspect) {
    const expanded = height * targetAspect;
    const centre = (minX + maxX) / 2;
    width = expanded;
    return { x: centre - width / 2, y: minY, width, height };
  }
  if (width / height > targetAspect * 1.12) {
    const expanded = width / targetAspect;
    const centre = (minY + maxY) / 2;
    height = expanded;
    return { x: minX, y: centre - height / 2, width, height };
  }
  return { x: minX, y: minY, width, height };
}

function easeInOutCubic(value: number) {
  return value < 0.5 ? 4 * value * value * value : 1 - Math.pow(-2 * value + 2, 3) / 2;
}

export function KnowledgeGraph({ nodes, edges }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const cameraGroupRef = useRef<SVGGElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const draggedRef = useRef(false);
  const suppressClickRef = useRef(false);
  const cameraRef = useRef<Transform>({ x: 0, y: 0, scale: 1 });
  const animationFrameRef = useRef<number | null>(null);
  const wheelFrameRef = useRef<number | null>(null);
  const wheelDeltaRef = useRef(0);
  const wheelPointRef = useRef<{ x: number; y: number } | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);

  const viewBox = useMemo(() => calculateViewBox(nodes), [nodes]);
  const nodeById = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);

  const connected = useMemo(() => {
    if (!selectedNode) return new Set<string>();
    const ids = new Set<string>([selectedNode]);
    edges.forEach((edge) => {
      if (edge.source === selectedNode) ids.add(edge.target);
      if (edge.target === selectedNode) ids.add(edge.source);
    });
    return ids;
  }, [edges, selectedNode]);

  const selectedEdgeNodes = useMemo(() => {
    if (!selectedEdge) return new Set<string>();
    const edge = edges.find((candidate) => candidate.id === selectedEdge);
    return edge ? new Set([edge.source, edge.target]) : new Set<string>();
  }, [edges, selectedEdge]);

  const renderedEdges = useMemo(
    () => edges.flatMap((edge) => {
      const source = nodeById.get(edge.source);
      const target = nodeById.get(edge.target);
      if (!source || !target) return [];
      return [{ edge, geometry: edgeGeometry(source, target, edge.bend ?? 0) }];
    }),
    [edges, nodeById],
  );

  const setCamera = (next: Transform) => {
    cameraRef.current = next;
    cameraGroupRef.current?.setAttribute(
      "transform",
      `translate(${next.x} ${next.y}) scale(${next.scale})`,
    );
  };

  const cancelCameraAnimation = () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  };

  const animateCamera = (target: Transform, duration = 460) => {
    cancelCameraAnimation();
    const start = cameraRef.current;
    const startTime = performance.now();
    const tick = (now: number) => {
      const progress = Math.min(1, (now - startTime) / duration);
      const eased = easeInOutCubic(progress);
      setCamera({
        x: start.x + (target.x - start.x) * eased,
        y: start.y + (target.y - start.y) * eased,
        scale: start.scale + (target.scale - start.scale) * eased,
      });
      if (progress < 1) {
        animationFrameRef.current = window.requestAnimationFrame(tick);
      } else {
        animationFrameRef.current = null;
      }
    };
    animationFrameRef.current = window.requestAnimationFrame(tick);
  };

  useEffect(() => () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current);
    }
    if (wheelFrameRef.current !== null) {
      window.cancelAnimationFrame(wheelFrameRef.current);
    }
  }, []);

  const focusOnNodes = (nodeIds: string[], maxScale = 1.62) => {
    const focusNodes = nodeIds.map((id) => nodeById.get(id)).filter((node): node is GraphNode => Boolean(node));
    if (focusNodes.length === 0) return;
    const minX = Math.min(...focusNodes.map((node) => node.x - node.r));
    const maxX = Math.max(...focusNodes.map((node) => node.x + node.r));
    const minY = Math.min(...focusNodes.map((node) => node.y - node.r));
    const maxY = Math.max(...focusNodes.map((node) => node.y + node.r));
    const clusterWidth = Math.max(1, maxX - minX + 120);
    const clusterHeight = Math.max(1, maxY - minY + 120);
    const fitScale = Math.min((viewBox.width * 0.88) / clusterWidth, (viewBox.height * 0.82) / clusterHeight);
    const scale = Math.max(0.35, Math.min(maxScale, fitScale));
    const centreX = (minX + maxX) / 2;
    const centreY = (minY + maxY) / 2;
    animateCamera({
      x: viewBox.x + viewBox.width / 2 - centreX * scale,
      y: viewBox.y + viewBox.height / 2 - centreY * scale,
      scale,
    });
  };

  const toSvgPoint = (clientX: number, clientY: number) => {
    const svg = svgRef.current;
    const matrix = svg?.getScreenCTM();
    if (!svg || !matrix) return { x: 0, y: 0 };
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const local = point.matrixTransform(matrix.inverse());
    return { x: local.x, y: local.y };
  };

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    event.preventDefault();
    cancelCameraAnimation();

    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? window.innerHeight : 1;
    wheelDeltaRef.current += event.deltaY * unit;
    wheelPointRef.current = toSvgPoint(event.clientX, event.clientY);

    if (wheelFrameRef.current !== null) return;

    wheelFrameRef.current = window.requestAnimationFrame(() => {
      const delta = wheelDeltaRef.current;
      const cursor = wheelPointRef.current;
      wheelDeltaRef.current = 0;
      wheelFrameRef.current = null;
      if (!cursor || delta === 0) return;

      const current = cameraRef.current;
      const factor = Math.exp(-delta * 0.0012);
      const scale = Math.min(6, Math.max(0.25, current.scale * factor));
      const worldX = (cursor.x - current.x) / current.scale;
      const worldY = (cursor.y - current.y) / current.scale;
      setCamera({ x: cursor.x - worldX * scale, y: cursor.y - worldY * scale, scale });
    });
  };

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    suppressClickRef.current = false;
    cancelCameraAnimation();
    const point = toSvgPoint(event.clientX, event.clientY);
    const current = cameraRef.current;
    dragRef.current = { startX: point.x, startY: point.y, baseX: current.x, baseY: current.y, captured: false };
    draggedRef.current = false;
  };

  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const point = toSvgPoint(event.clientX, event.clientY);
    const deltaX = point.x - drag.startX;
    const deltaY = point.y - drag.startY;
    if (Math.hypot(deltaX, deltaY) > 3) {
      draggedRef.current = true;
      if (!drag.captured) {
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.captured = true;
      }
    }
    const current = cameraRef.current;
    setCamera({ ...current, x: drag.baseX + deltaX, y: drag.baseY + deltaY });
  };

  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    if (draggedRef.current) suppressClickRef.current = true;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const handleNodeClick = (event: MouseEvent<SVGGElement>, nodeId: string) => {
    event.stopPropagation();
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    setSelectedEdge(null);
    setHoveredEdge(null);
    if (selectedNode === nodeId) {
      setSelectedNode(null);
      return;
    }
    setSelectedNode(nodeId);
    const neighbours = edges
      .filter((edge) => edge.source === nodeId || edge.target === nodeId)
      .flatMap((edge) => [edge.source, edge.target]);
    focusOnNodes([nodeId, ...neighbours]);
  };

  const handleEdgeClick = (event: MouseEvent<SVGPathElement>, edge: GraphEdge) => {
    event.stopPropagation();
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    setSelectedNode(null);
    setHoveredEdge(null);
    if (selectedEdge === edge.id) {
      setSelectedEdge(null);
      return;
    }
    setSelectedEdge(edge.id);
    focusOnNodes([edge.source, edge.target], 1.85);
  };

  const handleCanvasClick = () => {
    if (suppressClickRef.current) {
      suppressClickRef.current = false;
      return;
    }
    setSelectedNode(null);
    setSelectedEdge(null);
    setHoveredEdge(null);
  };

  const resetView = () => {
    setSelectedNode(null);
    setSelectedEdge(null);
    setHoveredEdge(null);
    animateCamera({ x: 0, y: 0, scale: 1 }, 520);
  };

  return (
    <div
      className="graph-canvas"
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onClick={handleCanvasClick}
      onDoubleClick={resetView}
    >
      <svg
        ref={svgRef}
        className="graph-svg"
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Interactive book discovery graph. Scroll to zoom, drag to pan, and select a node or connection to focus its neighbourhood."
      >
        <defs>
          <marker id="bookgraph-arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="10" markerHeight="10" orient="auto" markerUnits="userSpaceOnUse">
            <path d="M 1 1 L 10 6 L 1 11" fill="none" stroke="#625d55" strokeWidth="1.35" strokeLinecap="round" strokeLinejoin="round" />
          </marker>
        </defs>

        <g ref={cameraGroupRef} className="graph-camera" transform="translate(0 0) scale(1)">
          <g className="graph-edges">
            {renderedEdges.map(({ edge, geometry }, index) => {
              const nodeActive = selectedNode ? edge.source === selectedNode || edge.target === selectedNode : true;
              const edgeActive = selectedEdge ? edge.id === selectedEdge : nodeActive;
              const highlighted = selectedEdge === edge.id || hoveredEdge === edge.id;
              return (
                <g key={edge.id} opacity={edgeActive ? 1 : 0.13}>
                  <path
                    d={geometry.path}
                    className={`edge-line ${selectedEdge === edge.id ? "is-edge-selected" : ""}`}
                    markerEnd="url(#bookgraph-arrow)"
                  />
                  <path
                    d={geometry.path}
                    className="edge-hit"
                    role="button"
                    tabIndex={0}
                    aria-label={`${nodeById.get(edge.source)?.label ?? "Entity"} ${edge.relation} ${nodeById.get(edge.target)?.label ?? "entity"}`}
                    onClick={(event) => handleEdgeClick(event, edge)}
                    onMouseEnter={() => setHoveredEdge(edge.id)}
                    onMouseLeave={() => setHoveredEdge((current) => current === edge.id ? null : current)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        handleEdgeClick(event as unknown as MouseEvent<SVGPathElement>, edge);
                      }
                    }}
                    style={{ stroke: highlighted ? "rgba(42, 39, 34, 0.10)" : "transparent" }}
                  />
                </g>
              );
            })}
          </g>

          <g className="graph-nodes">
            {nodes.map((node, index) => {
              const active = selectedNode ? connected.has(node.id) : selectedEdge ? selectedEdgeNodes.has(node.id) : true;
              const selected = selectedNode === node.id || selectedEdgeNodes.has(node.id);
              return (
                <g
                  key={node.id}
                  className={`graph-node ${selected ? "is-selected" : ""}`}
                  opacity={active ? 1 : 0.17}
                  style={{ animationDelay: `${index * 14}ms` }}
                  onClick={(event) => handleNodeClick(event, node.id)}
                  role="button"
                  tabIndex={0}
                  aria-label={`${node.label}, ${node.kind}`}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      handleNodeClick(event as unknown as MouseEvent<SVGGElement>, node.id);
                    }
                  }}
                >
                  <title>{node.label}</title>
                  <path
                    d={sketchCirclePath(node.x, node.y, node.r, index + 1)}
                    fill={fillByKind[node.kind]}
                    stroke="#292724"
                    strokeWidth={selected ? 4 : 2.8}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="node-outline"
                  />
                  <path
                    d={sketchCirclePath(node.x + 1.2, node.y - 0.8, node.r - 2.5, index + 4)}
                    fill="none"
                    stroke="#292724"
                    strokeWidth="0.9"
                    opacity="0.28"
                    strokeLinecap="round"
                    pointerEvents="none"
                  />
                  <text
                    x={node.x}
                    y={node.y}
                    className="node-text"
                    style={{ fontSize: `${labelFontSize(node)}px` }}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    pointerEvents="none"
                  >
                    {node.lines.map((line, lineIndex) => (
                      <tspan
                        key={`${node.id}-${lineIndex}`}
                        x={node.x}
                        dy={lineIndex === 0 ? `${-((node.lines.length - 1) * 0.54)}em` : "1.08em"}
                      >
                        {line}
                      </tspan>
                    ))}
                  </text>
                </g>
              );
            })}
          </g>

          <g className="graph-edge-labels" pointerEvents="none">
            {renderedEdges.filter(({ edge }) => edge.showLabel !== false).map(({ edge, geometry }) => {
              const nodeActive = selectedNode ? edge.source === selectedNode || edge.target === selectedNode : true;
              const active = selectedEdge ? edge.id === selectedEdge : nodeActive;
              return (
                <g
                  key={`label-${edge.id}`}
                  transform={`translate(${geometry.midX} ${geometry.midY - 10}) rotate(${geometry.angle})`}
                  opacity={active ? 1 : 0.13}
                >
                  <text className="edge-label" textAnchor="middle" dominantBaseline="middle">{edge.relation}</text>
                </g>
              );
            })}
          </g>
        </g>
      </svg>
    </div>
  );
}
