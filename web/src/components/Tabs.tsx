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
      className="grid w-full grid-cols-1 gap-1 rounded-lg border border-cb-border bg-cb-bg p-1 sm:flex sm:w-auto sm:flex-wrap"
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
            className={`min-h-11 rounded-lg border px-4 py-2 text-sm font-medium transition-[color,background-color,border-color] duration-150 ${
              selected
                ? "border-cb-border bg-cb-surface text-cb-text"
                : "border-transparent text-cb-muted hover:text-cb-text"
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
