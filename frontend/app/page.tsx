"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import type { FormEvent } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { KnowledgeGraph } from "@/components/knowledge-graph";
import { searchBookGraph } from "@/lib/search";
import type { GraphEdge, GraphNode } from "@/lib/graph-types";

type Screen = "home" | "loading" | "graph";

export default function HomePage() {
  const [portalReady, setPortalReady] = useState(false);
  const [query, setQuery] = useState("");
  const [screen, setScreen] = useState<Screen>("home");
  const [nodes, setNodes] = useState<GraphNode[]>([]);
  const [edges, setEdges] = useState<GraphEdge[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    setPortalReady(true);
  }, []);

  const handleSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const value = query.trim();
    if (!value) {
      setError("Enter a word to explore.");
      return;
    }

    setError("");
    setScreen("loading");
    try {
      const result = await searchBookGraph(value);
      if (result.nodes.length < 2 || result.edges.length === 0) {
        throw new Error("No meaningful book connections found. Try another word.");
      }
      setNodes(result.nodes);
      setEdges(result.edges);
      setScreen("graph");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "No meaningful book connections found. Try another word.");
      setScreen("home");
    }
  };

  const goBack = () => {
    setScreen("home");
    setQuery("");
    setError("");
    setNodes([]);
    setEdges([]);
  };

  if (screen === "graph" || screen === "loading") {
    return (
      <>
        <main className="graph-screen">
          {screen === "loading" ? (
            <div className="graph-loading" aria-label="Loading graph"><span /></div>
          ) : (
            <KnowledgeGraph nodes={nodes} edges={edges} />
          )}
        </main>
        {portalReady ? createPortal(
          <button className="back-button" onClick={goBack} aria-label="Return to search" title="Return to search">
            <ArrowLeft size={20} strokeWidth={1.7} />
          </button>,
          document.body,
        ) : null}
      </>
    );
  }

  return (
    <main className="landing-screen">
      <section className="landing-content">
        <div className="wordmark" aria-label="BookGraph">Book<span>Graph</span></div>
        <p className="tagline">Discover the connections between books.</p>
        <form className="search-form" onSubmit={handleSearch}>
          <input
            aria-label="Enter a word, author or theme"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              if (error) setError("");
            }}
            placeholder="Enter a word..."
            autoComplete="off"
            spellCheck={false}
          />
          <button type="submit" aria-label="Explore graph"><ArrowRight size={21} strokeWidth={1.8} /></button>
        </form>
        {error ? <p className="search-error" role="alert">{error}</p> : null}
      </section>
    </main>
  );
}
