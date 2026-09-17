import type { ReactNode } from "react";
import { FlaskConical, Info } from "lucide-react";
import { DashboardTabs } from "@/components/dashboard/dashboard-tabs";

export const dynamic = "force-dynamic";

const GLOSSARY: Array<[string, string]> = [
  ["F1", "Balance of precision and recall. The headline detection metric on the test split."],
  ["AUROC", "Probability that a random positive case scores above a random negative one."],
  ["PR-AUC", "Precision-recall area. More informative than AUROC when classes are imbalanced."],
  ["MCC", "Matthews correlation coefficient. A balanced single-number quality score."],
  ["ECE", "Expected calibration error. Gap between predicted probabilities and observed frequencies."],
  ["Brier", "Mean squared error of the predicted probabilities. Lower is better."],
  ["Platt / isotonic", "Post-hoc calibration methods. Platt is the deployable default here."],
  ["SHAP", "Per-feature attribution explaining how the model reached a score."],
  ["McNemar", "Paired significance test between two classifiers on the same test cases."],
  ["Bootstrap CI", "Confidence interval from resampling with replacement."],
];

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <div className="space-y-6">
      <div className="glass-panel rounded-2xl p-5 md:p-6 animate-fade">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl surface-inset">
              <FlaskConical className="h-5 w-5 text-violet-600 dark:text-purple-400" aria-hidden />
            </div>
            <div>
              <p className="eyebrow">Version B evidence</p>
              <h1 className="display-title text-2xl">Experiment dashboard</h1>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Every value is read from <code className="font-mono">artifacts/results</code> at request time, never hardcoded.
              </p>
            </div>
          </div>
        </div>

        <details className="mt-4 border-t border-border/40 pt-3">
          <summary className="flex cursor-pointer items-center gap-1.5 text-[11px] font-semibold text-muted-foreground hover:text-foreground">
            <Info className="h-3.5 w-3.5" aria-hidden />
            How to read these metrics
          </summary>
          <dl className="mt-3 grid gap-x-6 gap-y-2 sm:grid-cols-2">
            {GLOSSARY.map(([term, def]) => (
              <div key={term} className="text-[11px] leading-relaxed">
                <dt className="font-semibold text-foreground">{term}</dt>
                <dd className="text-muted-foreground">{def}</dd>
              </div>
            ))}
          </dl>
        </details>
      </div>

      <DashboardTabs />
      {children}
    </div>
  );
}
