"use client";

import { useEffect, useState, type ComponentPropsWithoutRef, type JSX } from "react";
import {
  MarkdownTextPrimitive,
  useIsMarkdownCodeBlock,
  type CodeHeaderProps,
  type SyntaxHighlighterProps,
} from "@assistant-ui/react-markdown";
import { Check, Copy } from "lucide-react";
import { createHighlighter } from "shiki";

let highlighterPromise: ReturnType<typeof createHighlighter> | null = null;

function getHighlighter() {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({
      themes: ["github-light", "github-dark"],
      langs: ["json", "python", "javascript", "typescript", "bash", "sql", "yaml", "markdown", "text"],
    });
  }
  return highlighterPromise;
}

function useIsDark() {
  const [isDark, setIsDark] = useState(
    () => typeof document !== "undefined" && document.documentElement.classList.contains("dark")
  );
  useEffect(() => {
    const el = document.documentElement;
    const observer = new MutationObserver(() => {
      setIsDark(el.classList.contains("dark"));
    });
    observer.observe(el, { attributes: true, attributeFilter: ["class"] });
    return () => observer.disconnect();
  }, []);
  return isDark;
}

function ShikiCodeBlock({ language, code }: SyntaxHighlighterProps) {
  const isDark = useIsDark();
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHighlighter().then((highlighter) => {
      if (cancelled) return;
      setHtml(
        highlighter.codeToHtml(code, {
          lang: language === "unknown" ? "text" : language || "text",
          theme: isDark ? "github-dark" : "github-light",
        })
      );
    });
    return () => {
      cancelled = true;
    };
  }, [code, language, isDark]);

  if (!html) {
    return (
      <pre className="my-3 overflow-x-auto rounded-b-xl border border-t-0 border-border bg-muted/50 p-3 text-[13px] leading-relaxed">
        <code>{code}</code>
      </pre>
    );
  }

  return (
    <div
      className="shiki-shell my-3 overflow-x-auto rounded-b-xl border border-t-0 border-border bg-muted/50 p-3 text-[13px] leading-relaxed"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

function CodeHeader({ language, code }: CodeHeaderProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch (error) {
      console.error("Failed to copy code:", error);
    }
  };

  return (
    <div className="flex items-center justify-between gap-2 rounded-t-xl border border-b-0 border-border bg-muted/70 px-3 py-1.5">
      <span className="truncate font-mono text-[10px] uppercase tracking-wider text-muted-foreground">
        {language === "unknown" ? "code" : language}
      </span>
      <button
        type="button"
        onClick={handleCopy}
        aria-label="Copy code"
        className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
      >
        {copied ? <Check className="h-3 w-3 text-emerald-500" /> : <Copy className="h-3 w-3" />}
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

type MarkdownComponentProps<T extends keyof JSX.IntrinsicElements> =
  ComponentPropsWithoutRef<T> & { node?: unknown };

function stripNode<T extends object>(props: T) {
  const { node, ...rest } = props as T & { node?: unknown };
  void node;
  return rest;
}

function InlineCode(props: MarkdownComponentProps<"code">) {
  if (useIsMarkdownCodeBlock()) {
    return <code {...stripNode(props)} />;
  }
  return (
    <code
      {...stripNode(props)}
      className="rounded-md bg-muted/70 px-1.5 py-0.5 font-mono text-[0.85em] text-foreground"
    />
  );
}

export function MarkdownText() {
  return (
    <MarkdownTextPrimitive
      className="min-w-0 text-sm leading-relaxed"
      defer
      components={{
        h1: (props: MarkdownComponentProps<"h1">) => (
          <h1 className="mb-2 mt-1 text-base font-bold tracking-tight" {...stripNode(props)} />
        ),
        h2: (props: MarkdownComponentProps<"h2">) => (
          <h2 className="gradient-text mb-2 mt-4 text-sm font-bold" {...stripNode(props)} />
        ),
        h3: (props: MarkdownComponentProps<"h3">) => (
          <h3 className="mb-1.5 mt-3 text-sm font-semibold" {...stripNode(props)} />
        ),
        p: (props: MarkdownComponentProps<"p">) => (
          <p className="mb-2 last:mb-0" {...stripNode(props)} />
        ),
        strong: (props: MarkdownComponentProps<"strong">) => (
          <strong className="font-semibold text-foreground" {...stripNode(props)} />
        ),
        em: (props: MarkdownComponentProps<"em">) => <em {...stripNode(props)} />,
        ul: (props: MarkdownComponentProps<"ul">) => (
          <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0" {...stripNode(props)} />
        ),
        ol: (props: MarkdownComponentProps<"ol">) => (
          <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0" {...stripNode(props)} />
        ),
        li: (props: MarkdownComponentProps<"li">) => (
          <li className="leading-relaxed" {...stripNode(props)} />
        ),
        a: (props: MarkdownComponentProps<"a">) => (
          <a
            className="font-medium text-primary underline underline-offset-2"
            target="_blank"
            rel="noreferrer"
            {...stripNode(props)}
          />
        ),
        blockquote: (props: MarkdownComponentProps<"blockquote">) => (
          <blockquote
            className="mb-2 border-l-2 border-primary/50 pl-3 italic text-muted-foreground last:mb-0"
            {...stripNode(props)}
          />
        ),
        hr: (props: MarkdownComponentProps<"hr">) => (
          <hr className="my-3 border-border" {...stripNode(props)} />
        ),
        pre: (props: MarkdownComponentProps<"pre">) => (
          <pre
            className="my-3 overflow-x-auto rounded-xl border border-border bg-muted/50 p-3 text-[13px] leading-relaxed"
            {...stripNode(props)}
          />
        ),
        code: InlineCode,
        table: (props: MarkdownComponentProps<"table">) => (
          <div className="my-2 overflow-x-auto">
            <table className="w-full border-collapse text-sm" {...stripNode(props)} />
          </div>
        ),
        thead: (props: MarkdownComponentProps<"thead">) => (
          <thead className="border-b border-border bg-muted/50" {...stripNode(props)} />
        ),
        th: (props: MarkdownComponentProps<"th">) => (
          <th className="px-3 py-2 text-left text-xs font-semibold text-muted-foreground" {...stripNode(props)} />
        ),
        td: (props: MarkdownComponentProps<"td">) => (
          <td className="border-b border-border/60 px-3 py-2 align-top" {...stripNode(props)} />
        ),
        tr: (props: MarkdownComponentProps<"tr">) => <tr {...stripNode(props)} />,
        CodeHeader,
        SyntaxHighlighter: ShikiCodeBlock,
      }}
    />
  );
}
