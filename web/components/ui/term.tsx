import type { ReactNode } from "react";

/** Dotted-underline abbreviation with a native tooltip; works without JS. */
export function Term({ children, def }: { children: ReactNode; def: string }) {
  return (
    <abbr
      title={def}
      className="cursor-help underline decoration-dotted underline-offset-2 decoration-muted-foreground/50 no-underline"
    >
      {children}
    </abbr>
  );
}
