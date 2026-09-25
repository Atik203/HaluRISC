import { NavBar } from "@/components/nav-bar";
import { Footer } from "@/components/footer";
import { readJson } from "@/lib/results";

export default function SiteLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Server-side: version badge comes from the frozen manifest, never hardcoded.
  const manifest = readJson<{ model_version?: string | null }>("manifest.json");
  const version = manifest?.model_version ?? null;

  return (
    <>
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
    </>
  );
}
