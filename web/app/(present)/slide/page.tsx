import type { Metadata } from "next";
import { SlideDeck } from "@/components/slides/deck";
import { buildSlideData } from "@/lib/slide-data";

export const metadata: Metadata = {
  title: "HaluRISC — Presentation",
  robots: { index: false },
};

export default function SlidePage() {
  return <SlideDeck data={buildSlideData()} />;
}
