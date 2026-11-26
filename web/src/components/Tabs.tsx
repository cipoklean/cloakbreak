import { useRef, type KeyboardEvent } from "react";

export type TabId = "samples" | "upload" | "live";

const TAB_DEFS: { id: TabId; label: string }[] = [
  { id: "samples", label: "Try samples" },
  { id: "upload", label: "Upload or paste a link" },
  { id: "live", label: "Live mic (experimental)" },
];

interface TabsProps {
  activeTab: TabId;
  onChange: (tab: TabId) => void;
}

export function Tabs({ activeTab, onChange }: TabsProps) {
  const tabRefs = useRef<Record<TabId, HTMLButtonElement | null>>({
    samples: null,
    upload: null,
    live: null,
  });

  const activate = (id: TabId, focus = false) => {
    onChange(id);
    if (focus) window.setTimeout(() => tabRefs.current[id]?.focus(), 0);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = TAB_DEFS.findIndex((tab) => tab.id === activeTab);
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % TAB_DEFS.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + TAB_DEFS.length) % TAB_DEFS.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = TAB_DEFS.length - 1;
    else return;
    event.preventDefault();
    activate(TAB_DEFS[next].id, true);
  };

  return (
    <div
      role="tablist"
      aria-label="Input source"
      className="inline-flex max-w-full flex-wrap gap-1 rounded-xl border border-[#d5e2f0] bg-slate-100 p-1"
      onKeyDown={onKeyDown}
    >
      {TAB_DEFS.map((tab) => {
        const selected = tab.id === activeTab;
        return (
          <button
            key={tab.id}
            ref={(node) => {
              tabRefs.current[tab.id] = node;
            }}
            type="button"
            role="tab"
            id={`tab-${tab.id}`}
            aria-controls={`panel-${tab.id}`}
            aria-selected={selected}
            tabIndex={selected ? 0 : -1}
            data-testid={`tab-${tab.id}`}
            className={`min-h-11 rounded-lg px-4 py-2 text-sm font-semibold transition duration-150 hover:text-slate-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0369a1] focus-visible:ring-offset-2 ${
              selected ? "bg-white text-slate-950 shadow-sm" : "text-slate-600"
            }`}
            onClick={() => activate(tab.id)}
          >
            {tab.label}
          </button>
        );
      })}
    </div>
  );
}
