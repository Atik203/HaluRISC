import Link from "next/link";
import { ShieldCheck } from "lucide-react";

const LINKS = [
  { href: "/chat", label: "Chat" },
  { href: "/analyze", label: "Analyze" },
  { href: "/dashboard/overview", label: "Dashboard" },
  { href: "/demo", label: "Presenter demo" },
  { href: "/about", label: "About" },
];

export function Footer() {
  return (
    <footer className="border-t border-border/40 mt-10">
      <div className="max-w-[88rem] mx-auto w-full px-4 md:px-6 py-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-500 flex items-center justify-center text-white">
            <ShieldCheck className="w-4 h-4" aria-hidden />
          </div>
          <div>
            <p className="text-sm font-semibold leading-none">HaluRISC</p>
            <p className="text-[11px] text-muted-foreground mt-1">
              Calibrated, explainable hallucination risk for black-box LLM answers
            </p>
          </div>
        </div>

        <nav aria-label="Footer" className="flex flex-wrap items-center gap-x-5 gap-y-2">
          {LINKS.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
            >
              {l.label}
            </Link>
          ))}
        </nav>
      </div>
    </footer>
  );
}
