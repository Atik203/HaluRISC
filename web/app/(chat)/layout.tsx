export default function ChatLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // The chat route is a full-viewport application shell: no global navbar or
  // footer, its own scrolling regions, and a floating composer.
  return <div className="flex h-dvh min-h-0 flex-col overflow-hidden">{children}</div>;
}
