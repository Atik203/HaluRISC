"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  MessageSquare,
  BarChart2,
  LayoutDashboard,
  Info,
  ShieldCheck,
  Menu,
  X,
  Presentation,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { ProjectorToggle } from "@/components/projector-toggle";

const NAV_ITEMS = [
  { href: "/chat", label: "Chat Mode", icon: MessageSquare },
  { href: "/analyze", label: "Analyze Mode", icon: BarChart2 },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/demo", label: "Presenter", icon: Presentation },
  { href: "/about", label: "About", icon: Info },
];

export function NavBar({ version }: { version?: string | null }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const isActive = (href: string) =>
    pathname === href || (href === "/dashboard" && pathname.startsWith("/dashboard/"));

  return (
    <header className="border-b border-border/40 bg-background">
      <div className="max-w-[88rem] mx-auto px-4 h-16 flex items-center justify-between">
        {/* Brand */}
        <Link
          href="/"
          aria-label="HaluRISC home"
          className="flex items-center gap-2.5 font-bold tracking-tight rounded-xl"
        >
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-500 flex items-center justify-center text-white shadow-lg shadow-violet-600/20">
            <ShieldCheck className="w-5 h-5" aria-hidden />
          </div>
          <span className="gradient-text display-title text-xl leading-none">HaluRISC</span>
          {version && (
            <span
              title={`Model version ${version} (from the frozen manifest)`}
              className="text-[10px] uppercase tracking-widest text-muted-foreground bg-secondary/80 px-2 py-0.5 rounded-full border border-border"
            >
              {version}
            </span>
          )}
        </Link>

        {/* Desktop navigation */}
        <div className="flex items-center gap-2">
          <nav
            aria-label="Main"
            className="hidden md:flex items-center gap-1 bg-secondary/50 p-1 rounded-full border border-border/50"
          >
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold transition-all duration-200 ${
                    active
                      ? "bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md shadow-violet-600/20"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                  }`}
                >
                  <Icon className="w-4 h-4" aria-hidden />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
          <ProjectorToggle />
          <ThemeToggle />

          {/* Mobile menu button */}
          <button
            type="button"
            className="md:hidden inline-flex items-center justify-center w-9 h-9 rounded-xl border border-border/60 text-muted-foreground hover:text-foreground transition-all"
            aria-label={open ? "Close navigation menu" : "Open navigation menu"}
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((v) => !v)}
          >
            {open ? <X className="w-5 h-5" aria-hidden /> : <Menu className="w-5 h-5" aria-hidden />}
          </button>
        </div>
      </div>

      {/* Mobile navigation */}
      {open && (
        <nav
          id="mobile-nav"
          aria-label="Main (mobile)"
          className="md:hidden border-t border-border/40 p-3 space-y-1 animate-fade"
        >
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setOpen(false)}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-semibold transition-all ${
                  active
                    ? "bg-gradient-to-r from-violet-600 to-indigo-600 text-white shadow-md shadow-violet-600/20"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                }`}
              >
                <Icon className="w-4 h-4" aria-hidden />
                {item.label}
              </Link>
            );
          })}
        </nav>
      )}
    </header>
  );
}
