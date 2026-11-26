import { useState, type ReactNode, type SyntheticEvent } from "react";
import { getJson } from "../api";
import type { FactsResponse } from "../types";
import { formatMeasuredNumber } from "../utils";

function inlineMarkdown(text: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*)/g).filter((part) => part.length > 0);
  return parts.map((part, index) => {
    if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
      return (
        <code key={index} className="rounded bg-slate-100 px-1 py-0.5 font-mono text-[0.9em] text-slate-900">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <span key={index}>{part}</span>;
  });
}

type MarkdownBlock =
  | { type: "heading"; level: number; text: string }
  | { type: "ordered"; items: { num: number; text: string }[] }
  | { type: "unordered"; items: { text: string; indent: number }[] }
  | { type: "paragraph"; text: string };

function parseMarkdown(markdown: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = [];
  let ordered: { num: number; text: string }[] | null = null;
  let unordered: { text: string; indent: number }[] | null = null;
  const flush = () => {
    if (ordered && ordered.length) blocks.push({ type: "ordered", items: ordered });
    if (unordered && unordered.length) blocks.push({ type: "unordered", items: unordered });
    ordered = null;
    unordered = null;
  };

  for (const line of markdown.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      flush();
      continue;
    }
    const heading = /^(#{2,4})\s+(.*)$/.exec(trimmed);
    if (heading) {
      flush();
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      continue;
    }
    const orderedMatch = /^(\d+)\.\s+(.*)$/.exec(trimmed);
    if (orderedMatch) {
      if (unordered) flush();
      if (!ordered) ordered = [];
      ordered.push({ num: Number(orderedMatch[1]), text: orderedMatch[2] });
      continue;
    }
    const bulletMatch = /^(\s*)[-*]\s+(.*)$/.exec(line);
    if (bulletMatch) {
      // Nested bullets end the surrounding ordered block, but flushing it here
      // preserves the item they belong to. Explicit li values keep numbering
      // correct when the next ordered item starts a new block afterwards.
      if (ordered) flush();
      if (!unordered) unordered = [];
      unordered.push({ text: bulletMatch[2], indent: Math.min(3, Math.floor(bulletMatch[1].length / 2)) });
      continue;
    }
    flush();
    blocks.push({ type: "paragraph", text: trimmed });
  }
  flush();
  return blocks;
}

export function SafeMarkdown({ markdown }: { markdown: string }) {
  const blocks = parseMarkdown(markdown);
  return (
    <div className="space-y-3 text-sm leading-relaxed text-slate-700" data-testid="honest-limits-markdown">
      {blocks.map((block, index) => {
        if (block.type === "heading") {
          return (
            <h3 key={index} className={`font-bold text-slate-950 ${block.level === 2 ? "text-lg" : "text-base"}`}>
              {inlineMarkdown(block.text)}
            </h3>
          );
        }
        if (block.type === "ordered") {
          return (
            <ol key={index} className="list-decimal space-y-2 pl-6">
              {block.items.map((item) => (
                <li key={item.num} value={item.num}>{inlineMarkdown(item.text)}</li>
              ))}
            </ol>
          );
        }
        if (block.type === "unordered") {
          return (
            <ul key={index} className="list-disc space-y-1.5 pl-6">
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex} style={{ marginLeft: `${item.indent}rem` }}>
                  {inlineMarkdown(item.text)}
                </li>
              ))}
            </ul>
          );
        }
        return <p key={index}>{inlineMarkdown(block.text)}</p>;
      })}
    </div>
  );
}

interface HonestLimitsProps {
  facts: FactsResponse | null;
}

export function HonestLimits({ facts }: HonestLimitsProps) {
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    if (markdown || loading) return;
    setLoading(true);
    setError(null);
    try {
      const body = await getJson<{ markdown: string }>("/api/honest-limits", "Could not load the limits section");
      setMarkdown(body.markdown);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the limits section");
    } finally {
      setLoading(false);
    }
  };

  const onToggle = (event: SyntheticEvent<HTMLDetailsElement>) => {
    if (event.currentTarget.open) void load();
  };

  const chips: string[] = [];
  if (facts && typeof facts.noise_10db_eer === "number") {
    chips.push(`Noisy audio: ${formatMeasuredNumber(facts.noise_10db_eer)}% error at 10 dB SNR`);
  }
  if (facts && (typeof facts.short_eer_0_5 === "number" || typeof facts.short_eer_0_8 === "number")) {
    chips.push("Clips under 1 s: at chance");
  }
  if (facts && typeof facts.headline_cb_eer === "number") {
    chips.push("Only one unseen cloner tested (Kokoro)");
  }
  if (facts && (typeof facts.livemic_frac_all === "number" || typeof facts.livemic_frac_gated === "number")) {
    chips.push("Live mic: false alarms on unfamiliar mics");
  }

  return (
    <section className="card rounded-2xl border border-[#d5e2f0] bg-white p-5 shadow-sm" data-testid="honest-limits-area">
      {chips.length ? (
        <div className="flex flex-wrap gap-2" aria-label="Honest limits">
          {chips.map((chip) => (
            <span key={chip} className="rounded-full border border-[#d5e2f0] bg-slate-50 px-3 py-1.5 text-sm text-slate-700" data-testid="honest-chip">
              {chip}
            </span>
          ))}
        </div>
      ) : null}
      <details className="group mt-4" onToggle={onToggle} data-testid="honest-limits-expander">
        <summary className="flex min-h-11 cursor-pointer list-none items-center rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-800 transition duration-150 hover:bg-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0369a1] focus-visible:ring-offset-2 [&::-webkit-details-marker]:hidden">
          <span>How it works and its limits</span>
          <span className="ml-auto text-slate-500 transition duration-150 group-open:rotate-90" aria-hidden="true">
            ›
          </span>
        </summary>
        <div className="mt-3 rounded-xl border border-[#d5e2f0] bg-slate-50 p-4">
          {loading ? <p className="text-sm text-slate-600">Loading limits…</p> : null}
          {error ? <p className="text-sm font-medium text-red-700">{error}</p> : null}
          {markdown ? <SafeMarkdown markdown={markdown} /> : null}
        </div>
      </details>
    </section>
  );
}
