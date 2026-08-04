"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageSquare, BarChart2, LayoutDashboard, Info, ShieldCheck } from "lucide-react";

export function NavBar() {
  const pathname = usePathname();

  const navItems = [
    { href: "/chat", label: "💬 Chat Mode", icon: MessageSquare },
    { href: "/analyze", label: "📊 Analyze Mode", icon: BarChart2 },
    { href: "/dashboard", label: "📈 Dashboard", icon: LayoutDashboard },
    { href: "/about", label: "ℹ️ About", icon: Info },
  ];

  return (
    <header className="sticky top-0 z-50 border-b border-border/40 glass-panel">
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <Link href="/" className="flex items-center gap-2 font-bold text-lg tracking-tight">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-violet-600 to-indigo-500 flex items-center justify-center text-white shadow-lg">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <span className="gradient-text text-xl">HaluRISC</span>
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground bg-secondary/80 px-2 py-0.5 rounded-full border border-border">
            v1.0
          </span>
        </Link>

        {/* Navigation Links */}
        <nav className="flex items-center gap-1 bg-secondary/50 p-1 rounded-xl border border-border/50">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href || (pathname === "/" && item.href === "/chat");
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all duration-200 ${
                  isActive
                    ? "bg-primary text-primary-foreground shadow-md"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary"
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
