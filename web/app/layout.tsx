import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { NavBar } from "@/components/nav-bar";
import { Footer } from "@/components/footer";
import { Toaster } from "@/components/ui/toast";
import { readJson } from "@/lib/results";

const instrument = localFont({
  src: [
    { path: "./fonts/instrument-sans-latin.woff2", weight: "400 700", style: "normal" },
    { path: "./fonts/instrument-sans-latin-italic.woff2", weight: "400 700", style: "italic" },
  ],
  variable: "--font-instrument",
  display: "swap",
  preload: true,
});

const newsreader = localFont({
  src: "./fonts/newsreader-latin.woff2",
  weight: "200 800",
  style: "normal",
  variable: "--font-newsreader",
  display: "swap",
  preload: false,
});

const jetbrains = localFont({
  src: "./fonts/jetbrains-mono-latin.woff2",
  weight: "100 800",
  style: "normal",
  variable: "--font-jetbrains",
  display: "swap",
  preload: false,
});

export const metadata: Metadata = {
  title: {
    default: "HaluRISC — Calibrated & Explainable Hallucination Risk Analyzer",
    template: "%s — HaluRISC",
  },
  description:
    "Lightweight, calibrated, and explainable ML framework for predicting hallucination risk in black-box LLM outputs.",
  applicationName: "HaluRISC",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0a0910" },
    { media: "(prefers-color-scheme: light)", color: "#f7f7fb" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Server-side: version badge comes from the frozen manifest, never hardcoded.
  const manifest = readJson<{ model_version?: string | null }>("manifest.json");
  const version = manifest?.model_version ?? null;

  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem("halurisc-theme");if(t==="light"){document.documentElement.classList.remove("dark")}else{document.documentElement.classList.add("dark")}}catch(e){document.documentElement.classList.add("dark")}try{if(localStorage.getItem("halurisc-projector")==="on"){document.documentElement.classList.add("projector")}}catch(e){}})();`,
          }}
        />
      </head>
      <body
        className={`${instrument.variable} ${newsreader.variable} ${jetbrains.variable} font-sans min-h-screen flex flex-col bg-background text-foreground antialiased`}
      >
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[100] focus:px-4 focus:py-2 focus:rounded-lg focus:bg-primary focus:text-primary-foreground"
        >
          Skip to main content
        </a>
        <NavBar version={version} />
        <main id="main-content" className="flex-1 max-w-[88rem] w-full mx-auto p-4 md:p-6">
          {children}
        </main>
        <Footer />
        <Toaster />
      </body>
    </html>
  );
}
