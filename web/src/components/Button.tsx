import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary";

const variantClasses: Record<ButtonVariant, string> = {
  primary:
    "border border-cb-text bg-cb-text text-cb-bg hover:bg-cb-secondary disabled:border-cb-muted disabled:bg-cb-muted disabled:text-cb-bg",
  secondary:
    "border border-cb-border bg-transparent text-cb-text hover:border-cb-muted hover:bg-cb-bg disabled:text-cb-muted",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  children: ReactNode;
}

export function Button({ variant = "secondary", className = "", children, type = "button", ...rest }: ButtonProps) {
  return (
    <button
      type={type}
      className={`inline-flex min-h-11 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-[color,background-color,border-color,transform] duration-150 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-55 disabled:active:translate-y-0 ${variantClasses[variant]} ${className}`}
      {...rest}
    >
      {children}
    </button>
  );
}
