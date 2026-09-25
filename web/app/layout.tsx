import type { Metadata, Viewport } from "next";
import Script from "next/script";
import localFont from "next/font/local";
import "./globals.css";
import { Toaster } from "@/components/ui/toast";

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
    { media: "(prefers-color-scheme: dark)", color: "#090813" },
    { media: "(prefers-color-scheme: light)", color: "#faf9fd" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        {/* HaluRISC ships its own dark theme; keep Dark Reader from injecting
            inline attributes that break React hydration. */}
        <meta name="darkreader-lock" />
      </head>
      <body
        className={`${instrument.variable} ${newsreader.variable} ${jetbrains.variable} font-sans min-h-screen flex flex-col bg-background text-foreground antialiased`}
      >
        {/* Applied before hydration to avoid a theme flash; Next hoists this
            into the document head (raw <script> in a component logs an error). */}
        <Script id="halurisc-theme-init" strategy="beforeInteractive">
          {`(function(){try{var t=localStorage.getItem("halurisc-theme");if(t==="light"){document.documentElement.classList.remove("dark")}else{document.documentElement.classList.add("dark")}}catch(e){document.documentElement.classList.add("dark")}try{if(localStorage.getItem("halurisc-projector")==="on"){document.documentElement.classList.add("projector")}}catch(e){}})();`}
        </Script>
        {children}
        <Toaster />
      </body>
    </html>
  );
}
