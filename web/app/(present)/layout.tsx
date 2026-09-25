export default function PresentLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // Presentation routes are full-screen and intentionally chrome-free.
  return <div className="h-dvh min-h-0 overflow-hidden">{children}</div>;
}
