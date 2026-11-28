interface IconProps {
  className?: string;
}

export function PlayIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M5.5 3.75v8.5c0 .62.69.99 1.2.65l6.1-4.25a.78.78 0 0 0 0-1.3L6.7 3.1c-.51-.34-1.2.03-1.2.65Z" fill="currentColor" />
    </svg>
  );
}

export function PauseIcon({ className = "h-4 w-4" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <rect x="4" y="3" width="2.6" height="10" rx="0.8" fill="currentColor" />
      <rect x="9.4" y="3" width="2.6" height="10" rx="0.8" fill="currentColor" />
    </svg>
  );
}
