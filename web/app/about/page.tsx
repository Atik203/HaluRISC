import { Info, Layers, Cpu, ShieldCheck } from "lucide-react";
import { featureCount, readJson } from "@/lib/results";

interface ManifestForAbout {
  model_version?: string | null;
  feature_version?: string | null;
  n_features?: number | null;
  versions?: Record<string, string | null>;
  hardware?: { cuda?: boolean; gpu_name?: string | null };
  seeds?: number[];
}

export default function AboutPage() {
  const manifest = readJson<ManifestForAbout>("manifest.json");
  const nFeatures = manifest?.n_features ?? featureCount() ?? 26;
  const versions = manifest?.versions ?? {};
  const tech = [
    "Next.js App Router",
    "assistant-ui",
    "Vercel AI SDK",
    "FastAPI",
    versions["scikit_learn"] ? `scikit-learn ${versions["scikit_learn"]}` : "scikit-learn",
    versions["xgboost"] ? `XGBoost ${versions["xgboost"]}` : "XGBoost",
    versions["torch"] ? `PyTorch ${versions["torch"]}` : "PyTorch",
    versions["shap"] ? `SHAP ${versions["shap"]}` : "SHAP",
    "Recharts",
    "Tailwind CSS v4",
  ];

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-2xl">
        <h1 className="text-2xl font-bold gradient-text flex items-center gap-2">
          <Info className="w-6 h-6 text-violet-600 dark:text-purple-400" /> About HaluRISC Framework
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          Calibrated, explainable hallucination risk prediction for black-box LLM outputs — Version B (B1–B5) evidence.
        </p>
      </div>

      {/* Methodology Section */}
      <div className="glass-panel p-6 rounded-2xl space-y-4">
        <h2 className="text-base font-bold flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-600 dark:text-indigo-400" /> Pipeline Architecture
        </h2>
        <p className="text-xs text-muted-foreground leading-relaxed">
          HaluRISC evaluates candidate LLM answers against provided context without inspecting model weights or activations.
          Features are computed across seven feature groups: length/style, lexical overlap, entity coverage, numeric consistency, hedging density, NLI contradiction, and semantic embeddings
          ({nFeatures} features total — read from feature_names.json).
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-purple-700 dark:text-purple-300 mb-1">1. Feature Extraction</h4>
            <p className="text-xs text-muted-foreground">
              {nFeatures} engineered features across 7 groups measuring grounding, entity coverage, NLI consistency, numeric novelty, and semantic drift (B1/B2).
            </p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-300 mb-1">2. Calibrated XGBoost</h4>
            <p className="text-xs text-muted-foreground">
              XGBoost with group-aware splits (no leakage) and Platt scaling; calibration-under-shift is measured in B4 — source calibration does not transfer, target calibration fixes it.
            </p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-blue-700 dark:text-blue-300 mb-1">3. SHAP Explanations</h4>
            <p className="text-xs text-muted-foreground">
              TreeExplainer attributions identifying features raising or lowering risk. B5 validates SHAP stability (top-1 never flips under controlled perturbations).
            </p>
          </div>
          <div className="bg-secondary/40 p-4 rounded-xl border border-border">
            <h4 className="text-xs font-bold uppercase tracking-wider text-amber-700 dark:text-amber-300 mb-1">4. Conversational AI UI</h4>
            <p className="text-xs text-muted-foreground">
              Next.js + assistant-ui for natural-language explanations with Generative UI; offline /demo walkthrough needs no API key.
            </p>
          </div>
        </div>
      </div>

      {/* Evidence Section */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h2 className="text-base font-bold flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400" /> Evidence (all numbers from artifacts)
        </h2>
        <p className="text-xs text-muted-foreground leading-relaxed">
          In-domain test F1 ≈ 0.98 with leakage-free grouped splits (B2) · zero-shot transfer drops on RAGTruth QA (AUROC ≈ 0.54)
          while FaithBench stays usable (F1 ≈ 0.81) (B3) · ECE 0.81 → 0.13 after target calibration (B4) · SHAP top-5 set
          Jaccard = 1.0 and top-1 feature never flips under perturbation (B5). Open the <a className="underline text-violet-600 dark:text-purple-400" href="/dashboard/overview">Dashboard</a> for the full tables.
        </p>
      </div>

      {/* Tech Stack List */}
      <div className="glass-panel p-6 rounded-2xl space-y-3">
        <h2 className="text-base font-bold flex items-center gap-2">
          <Cpu className="w-5 h-5 text-violet-600 dark:text-purple-400" /> Core Tech Stack
        </h2>
        <div className="flex flex-wrap gap-2 text-xs">
          {tech.map((t, i) => (
            <span key={i} className="bg-secondary px-3 py-1.5 rounded-lg border border-border text-foreground font-mono">
              {t}
            </span>
          ))}
        </div>
        <p className="text-[11px] text-muted-foreground">
          Versions above are read from manifest.json (the frozen Colab run environment)
          {manifest?.hardware?.cuda ? `; trained on ${manifest.hardware.gpu_name}` : ""}.
          {manifest?.seeds ? ` Seeds ${manifest.seeds.join("/")}.` : ""}
        </p>
      </div>
    </div>
  );
}
