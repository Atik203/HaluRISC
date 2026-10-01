import {
  AlertTriangle,
  Crown,
  Database,
  Gauge,
  GraduationCap,
  Layers,
  Lightbulb,
  ListChecks,
  Search,
  ShieldCheck,
  Sigma,
  Target,
  TrendingUp,
  Users,
  Workflow,
  Wrench,
} from "lucide-react";
import {
  ACCENT,
  AMBER,
  Bullet,
  Card,
  Chip,
  DEEP_INK,
  Figure,
  fmt,
  KpiStrip,
  ms,
  NEAR_BLACK,
  pct,
  ROSE,
  SLATE,
  SlideFrame,
  SlideShot,
  SlideTable,
  StackedBar,
  TEAL,
} from "@/components/slides/primitives";

/* ------------------------------------------------------------------ */
/* Data contract (assembled server-side from artifacts/, all real)     */
/* ------------------------------------------------------------------ */

export interface BaselineRow {
  key: string;
  label: string;
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  auroc?: number | null;
  prAuc?: number | null;
  mcc?: number | null;
  ece?: number | null;
  brier?: number | null;
  highlighted?: boolean;
}

export interface AblationRow {
  variant: string;
  label: string;
  nFeatures?: number;
  f1?: number | null;
  auroc?: number | null;
  mcc?: number | null;
  ece?: number | null;
  highlighted?: boolean;
}

export interface ShiftRow {
  datasetLabel: string;
  standard: { recall?: number | null; f1?: number | null; auroc?: number | null; flagged?: number | null; ece?: number | null };
  ecxgb: { recall?: number | null; f1?: number | null; auroc?: number | null; flagged?: number | null; ece?: number | null };
}

export interface TaskRow {
  label: string;
  nRows?: number | null;
  f1?: number | null;
  auroc?: number | null;
  flagged?: number | null;
}

export interface SlideData {
  modelVersion: string;
  featureVersion: string;
  nliModel: string;
  seedsText: string;
  nIter: number | null;
  baseFeatures: number;
  claimFeatures: number;
  totalFeatures: number;
  featureGroups: Record<string, string[]>;
  componentMap: Record<string, string>;
  baselines: BaselineRow[];
  ablation: AblationRow[];
  shift: ShiftRow[];
  taskBreakdown: TaskRow[];
  mcnemarP?: number | null;
  mcnemarBest?: string | null;
  xgbF1CiLo?: number | null;
  xgbF1CiHi?: number | null;
  displayMethod: string;
  calibrationRows: number | null;
  b4RawEce?: number | null;
  b4PlattEce?: number | null;
  b4IsoEce?: number | null;
  halRawEce?: number | null;
  halPlattEce?: number | null;
  displayRawEce?: number | null;
  displayRawBrier?: number | null;
  displayEce?: number | null;
  displayBrier?: number | null;
  sourceEce?: number | null;
  targetEce?: number | null;
  kendall?: number | null;
  jaccard?: number | null;
  entityDelta?: number | null;
  numberDelta?: number | null;
  irrelevantDelta?: number | null;
  latencyP50?: number | null;
  artifactMb?: number | null;
  costPer1k?: number | null;
  judgeF1?: number | null;
  judgeCostPer1k?: number | null;
  judgeModel?: string | null;
  judgeN?: number | null;
  judgeAcc?: number | null;
  judgePrec?: number | null;
  judgeRec?: number | null;
  judgeP50?: number | null;
  judgeP95?: number | null;
  judgeAgreement?: number | null;
  judgeMcnemarP?: number | null;
  judgeWrongRight?: number | null;
  judgeRightWrong?: number | null;
  xgbSameAcc?: number | null;
  leakFree?: boolean | null;
  leakSpanning?: number | null;
  labelBalanceText: string;
  modelJudgeF1?: number | null;
  strictAlphaText: string;
  strictFprText: string;
  strictRecallText: string;
  fpNote: string;
}

type SlideProps = { data: SlideData; index: number; total: number };

const BADGE_INTRO = { badge: "Introduction", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_MOTIVATION = { badge: "Motivation and Research Gap", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_OBJECTIVE = { badge: "Objective", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_DATA = { badge: "Dataset", badgeBg: "#ccfbf1", badgeColor: TEAL };
const BADGE_CONVENTIONAL = { badge: "Conventional Method", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_PROPOSED = { badge: "Proposed Method", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_FLOW = { badge: "Flow Diagram", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_SETUP = { badge: "Experimental Setup", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_RESULTS = { badge: "Results", badgeBg: "#fee2e2", badgeColor: ROSE };
const BADGE_APPLICATION = { badge: "Application", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_UI = { badge: "UI Demonstration", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_END = { badge: "Conclusion", badgeBg: "#ccfbf1", badgeColor: TEAL };

/* ───────────────────────────── 1 · Title ─────────────────────────── */

const MEMBERS = [
  { name: "Md. Atikur Rahaman", id: "0112310298", leader: true },
  { name: "Saiful Alam Sabbir", id: "0112310105", leader: false },
  { name: "MD. Miraz Ahamed", id: "0112310524", leader: false },
];

function TitleSlide({ index }: SlideProps) {
  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden bg-white px-[6.5cqw] pb-[2.2cqh] pt-[3.6cqh]">
      <div
        className="pointer-events-none absolute right-[-12cqw] top-[-20cqh] h-[56cqh] w-[56cqh] rounded-full"
        style={{ background: ACCENT, opacity: 0.07 }}
      />
      <div
        className="pointer-events-none absolute bottom-[-24cqh] left-[-10cqw] h-[44cqh] w-[44cqh] rounded-full"
        style={{ background: TEAL, opacity: 0.06 }}
      />
      <div className="relative flex min-h-0 flex-1 flex-col items-center justify-center gap-[4.5cqh]">
      <div className="relative text-center">
        <div
          className="inline-block rounded-full px-[2.6cqw] py-[0.8cqh] text-[2cqh] font-bold uppercase tracking-[0.16em]"
          style={{ background: ACCENT, color: "#ffffff" }}
        >
          CSE 4889 - Machine Learning · Section E · Team Phantom Devs
        </div>
        <div className="mt-[2.2cqh] text-[8.2cqh] font-extrabold leading-none" style={{ color: ACCENT }}>
          HaluRISC
        </div>
        <h1
          className="mx-auto mt-[1.1cqh] max-w-[80cqw] text-[3.05cqh] font-extrabold leading-snug"
          style={{ color: NEAR_BLACK }}
        >
          Evidence-Consistent XGBoost for Calibrated Hallucination Risk
          Estimation in LLM Answers
        </h1>
        <div
          className="mt-[1.4cqh] text-[2.2cqh] font-bold"
          style={{ color: DEEP_INK }}
        >
          United International University · Department of Computer Science and Engineering
        </div>
        <div className="mt-[1.4cqh] flex items-center justify-center gap-[1.2cqw]">
          <Chip color={TEAL} bg="#ccfbf1">
            <GraduationCap size="2.2cqh" className="mr-[0.5cqw] inline" />
            Supervisor: Ohidujjaman Tuhin
          </Chip>
          <Chip color={ACCENT} bg="#e0e7ff">
            Team: Phantom Devs
          </Chip>
          <Chip color={AMBER} bg="#fef3c7">
            Section: E
          </Chip>
        </div>
      </div>

      <div className="relative grid grid-cols-3 gap-[1.4cqw]">
        {MEMBERS.map((m) => (
          <div
            key={m.id}
            className="flex items-center gap-[1.2cqw] rounded-xl border-2 px-[1.8cqw] py-[1.5cqh]"
            style={{
              borderColor: m.leader ? ACCENT : "#cbd5e1",
              background: m.leader ? "#eef2ff" : "#f8fafc",
            }}
          >
            <div
              className="flex flex-shrink-0 items-center justify-center rounded-full"
              style={{ width: "5cqh", height: "5cqh", background: m.leader ? ACCENT : "#334155" }}
            >
              {m.leader ? (
                <Crown size="2.8cqh" color="#ffffff" />
              ) : (
                <Users size="2.6cqh" color="#ffffff" />
              )}
            </div>
            <div className="min-w-0">
              <div className="text-[2.6cqh] font-bold leading-tight" style={{ color: NEAR_BLACK }}>
                {m.name}
              </div>
              <div
                className="text-[2.05cqh] font-semibold tracking-wide"
                style={{ color: m.leader ? ACCENT : SLATE }}
              >
                {m.id}
              </div>
            </div>
          </div>
        ))}
      </div>
      </div>

      <div
        className="relative flex items-center justify-end border-t-2 pt-[0.9cqh] text-[1.7cqh] font-bold"
        style={{ borderColor: "#e2e8f0", color: SLATE }}
      >
        <span className="tnum">{index + 1}</span>
      </div>
    </div>
  );
}

/* ────────────────────────── 2 · The problem ─────────────────────── */

function ProblemSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_INTRO}
      title="LLMs answer fluently. Some answers are wrong."
      subtitle="A confident wrong answer is hard to catch, and the model weights are not available."
      accent={ROSE}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-2 gap-[1.8cqw]">
        <SlideShot
          src="/slides/intro-answer-vs-evidence.png"
          alt="A document that states 5 days per decade, and a chat answer that says 10 days, joined by a red contradiction mark"
          caption="The evidence says 5 days per decade. The fluent answer says 10."
          color={ROSE}
          ratio="1.78"
        />
        <Card icon={<Lightbulb size="2.4cqh" color="#fff" />} title="Why it matters" color={AMBER} fill="#fffbeb">
          <ul>
            <Bullet>Judge models are slow and costly.</Bullet>
            <Bullet>Encoder classifiers need model weights and a GPU.</Bullet>
            <Bullet>Feature-based detectors rarely report calibration or shift.</Bullet>
            <Bullet>
              Target: a score that is <b>accurate, calibrated, explainable, fast</b>, with no
              access to model internals.
            </Bullet>
          </ul>
        </Card>
      </div>
      <div className="mt-[1.4cqh]">
        <div
          className="text-[1.95cqh] font-extrabold uppercase tracking-wide"
          style={{ color: SLATE }}
        >
          What the system returns
        </div>
        <div className="mt-[0.7cqh] flex items-center gap-[1cqw]">
          <Chip color={ROSE} bg="#fee2e2">
            Risk 47% · medium
          </Chip>
          <Chip color={ACCENT} bg="#e0e7ff">
            3 claims supported
          </Chip>
          <Chip color={AMBER} bg="#fef3c7">
            1 claim contradicted
          </Chip>
          <Chip color={TEAL} bg="#ccfbf1">
            SHAP: why the score moved
          </Chip>
          <span className="text-[2.05cqh] font-semibold" style={{ color: SLATE }}>
            61.8 ms, no model internals, {data.totalFeatures} features
          </span>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ───────────────────── 3 · Motivation & gaps ────────────────────── */

function MotivationSlide({ data, index, total }: SlideProps) {
  const gaps = [
    {
      icon: <Gauge size="2.4cqh" color="#fff" />,
      title: "Calibration",
      color: ACCENT,
      fill: "#eef2ff",
      lines: [
        "A risk score is only useful when its value matches the observed frequency.",
        "Lightweight detectors rarely report ECE or reliability diagrams.",
      ],
    },
    {
      icon: <Search size="2.4cqh" color="#fff" />,
      title: "Explanation quality",
      color: TEAL,
      fill: "#f0fdfa",
      lines: [
        "Token-level attributions are visualized, but their stability is not measured.",
        "We measure SHAP stability, neutralization, and perturbations.",
      ],
    },
    {
      icon: <Target size="2.4cqh" color="#fff" />,
      title: "Domain shift",
      color: AMBER,
      fill: "#fffbeb",
      lines: [
        "Detectors lose accuracy outside the training domain.",
        "Zero-shot RAGTruth and FaithBench runs expose the gap.",
      ],
    },
  ];
  const standard = data.shift[0]?.standard;
  const ecxgb = data.shift[0]?.ecxgb;
  return (
    <SlideFrame
      {...BADGE_MOTIVATION}
      title="Motivation and research gap"
      subtitle="Calibration, explanation reliability, and cross-domain behaviour, tested in one study."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.2fr_1fr] gap-[1.6cqw]">
        <SlideShot
          src="/slides/motivation-three-gaps.png"
          alt="Three panels: a gauge that disagrees with a bar chart, an attribution chart with an unstable bar, and a line that drops after crossing a domain boundary"
          caption="Calibration, explanation quality, and domain shift"
          color={ACCENT}
          ratio="1.78"
        />
        <div className="flex min-h-0 flex-col justify-center gap-[1.1cqh]">
          {gaps.map((gap) => (
            <div
              key={gap.title}
              className="rounded-xl border-2 px-[1.5cqw] py-[1.1cqh]"
              style={{ borderColor: gap.color, background: gap.fill }}
            >
              <div className="flex items-center gap-[0.8cqw]">
                <span
                  className="flex flex-shrink-0 items-center justify-center rounded-lg"
                  style={{ width: "3.6cqh", height: "3.6cqh", background: gap.color }}
                >
                  {gap.icon}
                </span>
                <span className="text-[2.1cqh] font-extrabold uppercase tracking-wide" style={{ color: gap.color }}>
                  {gap.title}
                </span>
              </div>
              <div className="mt-[0.5cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
                {gap.lines[0]}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="mt-[1.4cqh]">
        <div
          className="mb-[0.7cqh] text-[1.95cqh] font-extrabold uppercase tracking-wide"
          style={{ color: SLATE }}
        >
          What the study delivers
        </div>
        <KpiStrip
          items={[
            {
              value: `${pct(standard?.flagged, 1)} → ${pct(ecxgb?.flagged, 0)}`,
              label: "Flagged as risky · RAGTruth",
              color: ROSE,
            },
            {
              value: `${fmt(data.displayRawEce, 2)} → ${fmt(data.displayEce)}`,
              label: "Display ECE · RAGTruth QA",
              color: ACCENT,
            },
            { value: ms(data.latencyP50), label: "Median analysis time", color: TEAL },
          ]}
        />
      </div>
    </SlideFrame>
  );
}

/* ─────────────── 4 · Datasets (sources + in-domain) ─────────────── */

function DatasetsSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_DATA}
      title="Datasets: sources, target, and splits"
      subtitle="One in-domain benchmark for training, two public corpora that stay fully held out."
      accent={TEAL}
      index={index}
      total={total}
    >
      <div className="flex min-h-0 flex-1 flex-col gap-[1.8cqh]">
        <SlideTable
          head={["Corpus", "Source", "License", "Rows / groups", "Role in this study"]}
          headerBg="#ccfbf1"
          color={TEAL}
          widths={["17%", "29%", "14%", "16%", "24%"]}
          highlightRow={0}
          rows={[
            ["HaluEval QA", "Li et al. · EMNLP 2023", "MIT", "20,000 / 10,000", "Train · validate · test"],
            ["RAGTruth", "Niu et al. · ACL 2024", "MIT", "17,790 / 2,965", "Zero-shot transfer + calibration"],
            ["FaithBench", "Bao et al. · NAACL 2025", "CC BY-NC-SA", "750 / 750", "Summarization stress test"],
          ]}
        />
        <div className="grid min-h-0 flex-1 grid-cols-[1.05fr_1fr] gap-[1.8cqw]">
          <Card
            icon={<Database size="2.4cqh" color="#fff" />}
            title="HaluEval QA · in-domain"
            color={TEAL}
            fill="#f0fdfa"
          >
            <ul>
              <Bullet>
                <b>10,000 questions</b>, each with a passage and a pair of answers, giving{" "}
                <b>20,000 labeled rows</b>.
              </Bullet>
              <Bullet>
                Target is binary per answer: <b>1 = hallucinated</b>, <b>0 = grounded</b>, balanced{" "}
                {data.labelBalanceText} in every split.
              </Bullet>
              <Bullet>
                Terse answers mean style alone carries signal, which motivates the length features
                and the debiased variant.
              </Bullet>
            </ul>
          </Card>
          <div className="flex min-h-0 flex-col justify-center gap-[1.4cqh]">
            <SlideTable
              head={["Split", "Rows", "Groups", "Share"]}
              headerBg="#ccfbf1"
              color={TEAL}
              widths={["28%", "24%", "24%", "24%"]}
              rows={[
                ["Train", "14,000", "7,000", "70%"],
                ["Validation", "3,000", "1,500", "15%"],
                ["Test", "3,000", "1,500", "15%"],
              ]}
            />
            <StackedBar
              segments={[
                { label: "Train 70%", value: 70, color: TEAL },
                { label: "Val 15%", value: 15, color: ACCENT },
                { label: "Test 15%", value: 15, color: AMBER },
              ]}
            />
            <div
              className="rounded-lg px-[1.4cqw] py-[1cqh] text-[2.05cqh] font-bold leading-snug"
              style={{ background: "#ccfbf1", color: TEAL }}
            >
              Grouped split, verified leakage-free. Both answers of a question stay in one partition.
            </div>
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}




/* ────────────────────── 9 · Feature engineering ─────────────────── */

const GROUP_LABELS: Record<string, string> = {
  length: "Length & style",
  lexical: "Lexical overlap",
  entity: "Entity coverage",
  nli: "NLI consistency",
  numeric: "Numeric consistency",
  hedging: "Hedging",
  semantic: "Semantic drift",
  claim: "Claim-level aggregates",
};

const GROUP_MEASURES: Record<string, string> = {
  length: "Answer length and word shape",
  lexical: "Shared words with the context and the question",
  entity: "Named entities covered by the evidence",
  nli: "Entailment and contradiction, in both directions",
  numeric: "Numbers that match the evidence",
  hedging: "Hedge words and their density",
  semantic: "Embedding similarity to the context and question",
};

function ProposedSlide({ data, index, total }: SlideProps) {
  const groups = Object.entries(data.featureGroups);
  const baseGroups = groups.filter(([key]) => key !== "claim");
  const claimCount = data.featureGroups.claim?.length ?? data.claimFeatures;
  const changes = [
    { key: "m1", label: "M1 · Debias", color: AMBER },
    { key: "m2", label: "M2 · Claim features", color: TEAL },
    { key: "m3", label: "M3 · Multi-source · deployed", color: ACCENT },
  ];
  return (
    <SlideFrame
      {...BADGE_PROPOSED}
      title="Proposed method: evidence-consistent XGBoost"
      subtitle={`${data.baseFeatures} base features in seven groups, plus ${data.claimFeatures} claim-level aggregates and three cumulative changes.`}
      accent={AMBER}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.15fr_1fr] gap-[1.6cqw]">
        <SlideTable
          head={["Group", "What it measures", "Count"]}
          headerBg="#fef3c7"
          color={AMBER}
          widths={["28%", "56%", "16%"]}
          rows={baseGroups.map(([key, cols]) => [
            GROUP_LABELS[key] ?? key,
            GROUP_MEASURES[key] ?? "",
            String(cols.length),
          ])}
        />
        <div className="flex min-h-0 flex-col justify-center gap-[1.2cqh]">
          <Card
            icon={<Sigma size="2.4cqh" color="#fff" />}
            title={`Claim-level aggregates (${claimCount})`}
            color={TEAL}
            fill="#f0fdfa"
          >
            <ul>
              <Bullet>Max and mean contradiction, and the contradicted-claim ratio.</Bullet>
              <Bullet>Supported and unsupported claim ratios.</Bullet>
              <Bullet>Min and mean entailment, with verdicts support-first and a relevance gate.</Bullet>
            </ul>
          </Card>
          <div
            className="rounded-xl border-2 px-[1.6cqw] py-[1.1cqh]"
            style={{ borderColor: ACCENT, background: "#eef2ff" }}
          >
            <div className="text-[1.95cqh] font-extrabold uppercase tracking-wide" style={{ color: ACCENT }}>
              Atomic clause splitting
            </div>
            <div className="mt-[0.5cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
              &ldquo;The change is dominated by a later freezeup, <b>and</b> the region is at its warmest&hellip;&rdquo;
              splits into two claims, each checked on its own.
            </div>
          </div>
        </div>
      </div>
      <div className="mt-[1.2cqh] grid grid-cols-3 gap-[1.2cqw]">
        {changes.map((c) => (
          <div key={c.key} className="rounded-xl border-2 bg-white px-[1.4cqw] py-[1cqh]" style={{ borderColor: c.color }}>
            <div className="text-[2.05cqh] font-extrabold" style={{ color: c.color }}>
              {c.label}
            </div>
            <div className="mt-[0.4cqh] text-[1.95cqh] font-medium leading-snug" style={{ color: DEEP_INK }}>
              {data.componentMap[c.key] ?? "—"}
            </div>
          </div>
        ))}
      </div>
    </SlideFrame>
  );
}

/* ───────────────────────── 10 · Models ──────────────────────────── */

function ConventionalSlide({ data, index, total }: SlideProps) {
  const baselines = [
    { title: "Overlap heuristic", color: SLATE, lines: ["Risk = 1 − lexical overlap", "Threshold tuned on validation"] },
    { title: "Logistic regression", color: ACCENT, lines: ["Standardized features", "Linear decision boundary"] },
    { title: "Random forest", color: TEAL, lines: ["300 trees", "Strong tabular baseline"] },
    { title: "XGBoost (reference)", color: AMBER, lines: ["30-iteration randomized search", "Grouped 5-fold CV"] },
  ];
  return (
    <SlideFrame
      {...BADGE_CONVENTIONAL}
      title="Conventional method and baselines"
      subtitle="Four standard detectors, and the tuned XGBoost that the proposed model builds on."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid grid-cols-4 gap-[1.3cqw]">
        {baselines.map((baseline) => (
          <div
            key={baseline.title}
            className="rounded-xl border-2 px-[1.5cqw] py-[1.2cqh]"
            style={{ borderColor: baseline.color, background: "#ffffff" }}
          >
            <div className="text-[2.3cqh] font-extrabold" style={{ color: baseline.color }}>
              {baseline.title}
            </div>
            {baseline.lines.map((line) => (
              <div key={line} className="mt-[0.5cqh] text-[1.95cqh] font-medium" style={{ color: DEEP_INK }}>
                {line}
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="mt-[1.6cqh] grid min-h-0 flex-1 grid-cols-2 gap-[1.6cqw]">
        <Card icon={<Workflow size="2.4cqh" color="#fff" />} title="How the baseline works" color={ACCENT} fill="#eef2ff">
          <ul>
            <Bullet>Hand-crafted evidence features replace raw text for a tabular learner.</Bullet>
            <Bullet>Gradient-boosted trees add one tree at a time and fit the leftover error.</Bullet>
            <Bullet>A logistic link and Platt scaling turn the summed leaf scores into a probability.</Bullet>
          </ul>
        </Card>
        <Card icon={<AlertTriangle size="2.4cqh" color="#fff" />} title="Limits that motivate EC-XGB" color={AMBER} fill="#fffbeb">
          <ul>
            <Bullet>HaluEval is close to saturated, so extra features buy almost no in-domain F1.</Bullet>
            <Bullet>Length and overlap act as shortcuts, and the model leans on them.</Bullet>
            <Bullet>Under shift the raw score turns unreliable, and no claim-level signal reaches the classifier.</Bullet>
          </ul>
        </Card>
      </div>
      <div className="mt-[1.2cqh] flex flex-wrap items-center gap-[0.9cqw]">
        <Chip color={TEAL} bg="#ccfbf1">
          NLI backbone: {data.nliModel}
        </Chip>
        <Chip color={ACCENT} bg="#e0e7ff">
          Tuning: depth · learning rate · trees · subsample · colsample ({data.nIter ?? "—"} draws)
        </Chip>
      </div>
    </SlideFrame>
  );
}

/* ──────────────────────── 11 · Training protocol ─────────────────── */

function ProtocolSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_SETUP}
      title="Training and evaluation protocol"
      subtitle="Every number is the mean over three seeds, with grouped folds and validation-only calibration."
      accent={AMBER}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-4 gap-[1.3cqw]">
        <Card icon={<Layers size="2.2cqh" color="#fff" />} title="Splits" color={ACCENT} fill="#eef2ff">
          <div className="text-[2.2cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
            Grouped 70/15/15 by question.
            <br />
            Both answers of a question stay in one partition, so no leakage.
          </div>
        </Card>
        <Card icon={<Workflow size="2.2cqh" color="#fff" />} title="CV & tuning" color={TEAL} fill="#f0fdfa">
          <div className="text-[2.2cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
            Grouped 5-fold CV, {data.nIter ?? "—"}-iteration randomized search.
            <br />
            Selected: depth 4, lr 0.01, 500 trees, subsample 0.9, colsample 0.7.
          </div>
        </Card>
        <Card icon={<ListChecks size="2.2cqh" color="#fff" />} title="Seeds" color={AMBER} fill="#fffbeb">
          <div className="text-[2.2cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
            Seeds {data.seedsText}.
            <br />
            McNemar, bootstrap confidence intervals, and Wilcoxon signed-rank tests.
          </div>
        </Card>
        <Card icon={<Gauge size="2.2cqh" color="#fff" />} title="Calibration" color={ROSE} fill="#fff1f2">
          <div className="text-[2.2cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
            Calibrators fitted on validation only.
            <br />
            Strict operating point: test FPR {data.strictFprText} with recall {data.strictRecallText}.
          </div>
        </Card>
      </div>
      <div className="mt-[1.4cqh]">
        <div
          className="rounded-xl border-2 px-[1.6cqw] py-[1.2cqh]"
          style={{ borderColor: ACCENT, background: "#eef2ff" }}
        >
          <div className="text-[1.95cqh] font-extrabold uppercase tracking-wide" style={{ color: ACCENT }}>
            Machine and environment
          </div>
          <div className="mt-[0.7cqh] flex flex-wrap items-center gap-[0.7cqw]">
            <Chip color={ACCENT} bg="#ffffff">
              Windows 11 · Python 3.12 · RTX 3060 6 GB · CUDA 12.8 · 32 GB RAM
            </Chip>
            <Chip color={TEAL} bg="#ffffff">
              scikit-learn 1.9 · XGBoost 3.3 · SHAP 0.52 · spaCy 3.8 · sentence-transformers 5.6
            </Chip>
          </div>
        </div>
        <div className="mt-[1cqh] flex flex-wrap items-center gap-[0.8cqw]">
          {["F1", "AUROC", "PR-AUC", "MCC", "ECE", "Brier"].map((metric) => (
            <Chip key={metric} color={ACCENT} bg="#e0e7ff">
              {metric}
            </Chip>
          ))}
          <span className="text-[2cqh] font-semibold" style={{ color: SLATE }}>
            plus reliability diagrams and per-seed dispersion
          </span>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ───────────────────── 12 · Baseline comparison ─────────────────── */

function BaselineResultsSlide({ data, index, total }: SlideProps) {
  const mainKeys = ["heuristic_overlap", "lr_full", "rf_full", "xgboost"];
  const main = data.baselines.filter((b) => mainKeys.includes(b.key));
  const controls = data.baselines.filter((b) => !mainKeys.includes(b.key));
  return (
    <SlideFrame
      {...BADGE_RESULTS}
      title="Baseline comparison on HaluEval QA"
      subtitle="Test set, mean over seeds 42/123/456. ECE measures how reliable each model's risk score is."
      accent={ROSE}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1fr_1.1fr] gap-[1.6cqw]">
        <div className="flex min-h-0 flex-col justify-center gap-[1.1cqh]">
          <SlideTable
            head={["Model", "Precision", "Recall", "F1", "ECE"]}
            headerBg="#fee2e2"
            color={ROSE}
            widths={["34%", "16.5%", "16.5%", "16.5%", "16.5%"]}
            highlightRow={main.findIndex((b) => b.highlighted)}
            rows={main.map((b) => [b.label, fmt(b.precision), fmt(b.recall), fmt(b.f1), fmt(b.ece, 4)])}
          />
          <div
            className="rounded-lg px-[1.4cqw] py-[1cqh] text-[1.95cqh] font-semibold leading-snug"
            style={{ background: "#f1f5f9", color: DEEP_INK }}
          >
            Controls · {controls.map((b) => `${b.label} F1 ${fmt(b.f1)}`).join(" · ")}
          </div>
          <div
            className="rounded-lg px-[1.4cqw] py-[1cqh] text-[1.95cqh] font-bold leading-snug"
            style={{ background: "#fee2e2", color: ROSE }}
          >
            {data.fpNote || "XGBoost shows the fewest false positives at the same threshold."} F1 95% CI [
            {fmt(data.xgbF1CiLo, 4)}, {fmt(data.xgbF1CiHi, 4)}] · McNemar p ={" "}
            {data.mcnemarP != null ? data.mcnemarP.toFixed(3) : "—"}.
          </div>
        </div>
        <div className="flex min-h-0 flex-col items-center justify-center gap-[1cqh]">
          <Figure
            src="/api/figures/fig_roc_pr.png"
            alt="ROC and precision-recall curves on the HaluEval test set"
            height="48cqh"
          />
          <div
            className="rounded-lg border-2 px-[1.4cqw] py-[1cqh]"
            style={{ borderColor: ROSE, background: "#ffffff" }}
          >
            <div className="text-[2cqh] font-extrabold uppercase tracking-wide" style={{ color: ROSE }}>
              Why XGBoost was selected
            </div>
            <div className="mt-[0.45cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
              Best F1/AUROC balance, the lowest ECE among the learned models, and the fewest false alarms.
            </div>
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ───────────────────── 13 · EC-XGB results ──────────────────────── */

function EcXgbResultsSlide({ data, index, total }: SlideProps) {
  const standard = data.shift[0]?.standard;
  const ecxgb = data.shift[0]?.ecxgb;
  const m0 = data.ablation.find((a) => a.variant === "m0");
  const m3 = data.ablation.find((a) => a.highlighted);
  return (
    <SlideFrame
      {...BADGE_RESULTS}
      title="EC-XGB preserves accuracy and fixes shift over-flagging"
      subtitle="In-domain differences are not significant; the gain appears on natural, out-of-domain answers."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <KpiStrip
        items={[
          {
            value: `${pct(standard?.flagged, 1)} → ${pct(ecxgb?.flagged, 1)}`,
            label: "Flagged as risky · RAGTruth",
            color: ROSE,
          },
          { value: `${fmt(standard?.auroc)} → ${fmt(ecxgb?.auroc)}`, label: "AUROC · RAGTruth", color: TEAL },
          { value: `${fmt(standard?.ece)} → ${fmt(ecxgb?.ece)}`, label: "ECE · RAGTruth", color: ACCENT },
        ]}
      />
      <div className="mt-[1.4cqh] grid min-h-0 flex-1 grid-cols-[1fr_1.02fr] gap-[1.6cqw]">
        <div className="flex min-h-0 flex-col gap-[0.7cqh]">
          <div className="text-[1.95cqh] font-extrabold uppercase tracking-wide" style={{ color: SLATE }}>
            Standard vs EC-XGB on shifted corpora
          </div>
          <SlideTable
            head={["Corpus", "Model", "Recall", "F1", "Flagged"]}
            headerBg="#ccfbf1"
            color={TEAL}
            widths={["30%", "20%", "16%", "17%", "17%"]}
            rows={data.shift.flatMap((row) => [
              [row.datasetLabel, "Standard", fmt(row.standard.recall), fmt(row.standard.f1), pct(row.standard.flagged, 1)],
              [row.datasetLabel, "EC-XGB", fmt(row.ecxgb.recall), fmt(row.ecxgb.f1), pct(row.ecxgb.flagged, 1)],
            ])}
          />
          <div
            className="rounded-lg px-[1.3cqw] py-[0.7cqh] text-[1.95cqh] font-bold leading-snug"
            style={{ background: "#f0fdfa", color: TEAL }}
          >
            On unseen corpora, recall falls and precision rises as the flag rate collapses.
          </div>
          <div
            className="rounded-lg px-[1.3cqw] py-[0.7cqh] text-[1.95cqh] font-semibold leading-snug"
            style={{ background: "#eef2ff", color: ACCENT }}
          >
            In-domain ablation: m0 {fmt(m0?.f1, 4)} to m3 {fmt(m3?.f1, 4)} F1, not significant.
          </div>
        </div>
        <div className="flex min-h-0 items-center justify-center">
          <Figure
            src="/slides/ecxgb-shift.png"
            alt="Grouped bars of the flag rate and AUROC for the standard model and EC-XGB on RAGTruth, RAGTruth QA, and FaithBench"
            caption="Flag rate collapses while AUROC rises on shifted corpora"
            height="48cqh"
          />
        </div>
      </div>
    </SlideFrame>
  );
}



/* ──────────────── 14 · Interface: Chat and Analyze ──────────────── */

function ChatAnalyzeSlide({ index, total }: SlideProps) {
  const notes = [
    { title: "Input interface", color: ACCENT, fill: "#eef2ff", line: "Question, context, and answer boxes with one-click example presets." },
    { title: "Result display", color: TEAL, fill: "#f0fdfa", line: "Calibrated gauge with band markers, then the per-claim verdicts." },
    { title: "Evidence quotes", color: AMBER, fill: "#fffbeb", line: "Every verdict shows the context sentence it was checked against." },
    { title: "Thresholds", color: ROSE, fill: "#fff1f2", line: "Bands are printed on the gauge, so no lookup table is needed." },
  ];
  return (
    <SlideFrame
      {...BADGE_UI}
      title="Interface: Chat and Analyze"
      subtitle="Chat scores the answer while it streams. Analyze exposes the inputs, the calibrated gauge, and the band thresholds."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="flex min-h-0 flex-1 flex-col justify-center gap-[1.6cqh]">
        <div className="grid grid-cols-[2.11fr_2.06fr] gap-[1.3cqw]">
          <SlideShot
            src="/slides/chat.png"
            alt="Chat page: question, streamed answer, SHAP key contributors, and the evidence-score risk card with claim verdicts"
            caption="Chat · automatic risk card with claim verdicts"
            color={ACCENT}
            ratio="2.11"
          />
          <SlideShot
            src="/slides/analyze-mode.png"
            alt="Analyze page: question, context and answer inputs, one-click example sweep, and the calibrated gauge with band markers"
            caption="Analyze · inputs, example sweep, calibrated gauge"
            color={TEAL}
            ratio="2.06"
          />
        </div>
        <div
          className="rounded-lg px-[1.4cqw] py-[1cqh] text-[2.05cqh] font-semibold leading-snug"
          style={{ background: "#f1f5f9", color: DEEP_INK }}
        >
          The chat card leads with the verdict and the evidence score, then the SHAP contributors.
          Analyze runs the same pipeline with full input control, so a score of 35% reads as medium
          risk at a glance.
        </div>
        <div className="grid grid-cols-4 gap-[1.2cqw]">
          {notes.map((item) => (
            <div
              key={item.title}
              className="rounded-xl border-2 px-[1.4cqw] py-[1.1cqh]"
              style={{ borderColor: item.color, background: item.fill }}
            >
              <div className="text-[2.05cqh] font-extrabold uppercase tracking-wide" style={{ color: item.color }}>
                {item.title}
              </div>
              <div className="mt-[0.45cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
                {item.line}
              </div>
            </div>
          ))}
        </div>
      </div>
    </SlideFrame>
  );
}

/* ───────── 15 · Interface: explanations and comparison ───────── */

function ExplainCompareSlide({ index, total }: SlideProps) {
  const notes = [
    { title: "Explanation", color: ACCENT, fill: "#eef2ff", line: "SHAP bars show which features pushed the raw score up or down." },
    { title: "Transparency", color: TEAL, fill: "#f0fdfa", line: "The full table lists all 35 features and their values for the answer." },
    { title: "Ease of use", color: AMBER, fill: "#fffbeb", line: "One dark theme across chat, analyze, and dashboard, with projector mode." },
  ];
  return (
    <SlideFrame
      {...BADGE_UI}
      title="Interface: explanations and model comparison"
      subtitle="The score is never a black box. SHAP bars, the feature table, and every baseline are visible in one screen."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="flex min-h-0 flex-1 flex-col justify-center gap-[1.5cqh]">
        <div className="grid grid-cols-[1.29fr_1.21fr_1.57fr] gap-[1.3cqw]">
          <SlideShot
            src="/slides/analyze-shap.png"
            alt="SHAP feature contribution chart for the raw XGBoost score"
            caption="SHAP contributions · raw score"
            color={ACCENT}
            ratio="1.29"
          />
          <SlideShot
            src="/slides/chat-details.png"
            alt="Why-this-score panel: SHAP bars plus the full 35-feature table with values"
            caption="Why this score · 35-feature table"
            color={TEAL}
            ratio="1.21"
          />
          <SlideShot
            src="/slides/analyze-compare.png"
            alt="Model comparison card: deployed calibrated score against EC-XGB, standard XGBoost, random forest, logistic regression, and the overlap heuristic"
            caption="Model comparison · EC-XGB vs baselines"
            color={ROSE}
            ratio="1.57"
          />
        </div>
        <div className="grid grid-cols-3 gap-[1.2cqw]">
          {notes.map((item) => (
            <div
              key={item.title}
              className="rounded-xl border-2 px-[1.4cqw] py-[1.1cqh]"
              style={{ borderColor: item.color, background: item.fill }}
            >
              <div className="text-[2.05cqh] font-extrabold uppercase tracking-wide" style={{ color: item.color }}>
                {item.title}
              </div>
              <div className="mt-[0.45cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
                {item.line}
              </div>
            </div>
          ))}
        </div>
      </div>
    </SlideFrame>
  );
}

/* ──────────────────────── 19 · Conclusion ───────────────────────── */

function ConclusionSlide({ data, index }: SlideProps) {
  const ecxgb = data.shift[0]?.ecxgb;
  return (
    <SlideFrame
      {...BADGE_END}
      title="What we found, and what remains open"
      subtitle="A light, calibrated, explainable detector that holds up in-domain and flags far fewer natural answers."
      accent={TEAL}
      index={index}
    >
      <div className="grid min-h-0 flex-1 grid-cols-3 gap-[1.6cqw]">
        <Card icon={<ShieldCheck size="2.2cqh" color="#fff" />} title="Findings" color={TEAL} fill="#f0fdfa">
          <ul>
            <Bullet>
              In-domain F1 <b>{fmt(data.ablation.find((a) => a.highlighted)?.f1 ?? null, 4)}</b> with
              calibration intact.
            </Bullet>
            <Bullet>
              Shift over-flagging falls from {pct(data.shift[0]?.standard.flagged, 1)} to{" "}
              {pct(ecxgb?.flagged, 1)}.
            </Bullet>
            <Bullet>
              Display ECE drops from {fmt(data.displayRawEce)} to {fmt(data.displayEce)}.
            </Bullet>
          </ul>
        </Card>
        <Card icon={<AlertTriangle size="2.2cqh" color="#fff" />} title="Limitations" color={AMBER} fill="#fffbeb">
          <ul>
            <Bullet>English-only models and features.</Bullet>
            <Bullet>Held-out RAGTruth QA improves mainly in calibration, not F1.</Bullet>
            <Bullet>On FaithBench the model becomes conservative and trades F1 for a much lower flag rate.</Bullet>
          </ul>
        </Card>
        <Card icon={<Wrench size="2.2cqh" color="#fff" />} title="Future work" color={ACCENT} fill="#eef2ff">
          <ul>
            <Bullet>Multicalibration under stronger shift.</Bullet>
            <Bullet>Per-domain threshold transfer.</Bullet>
            <Bullet>Neural and verifier baselines beside EC-XGB.</Bullet>
          </ul>
        </Card>
      </div>
      <div className="mt-[1.4cqh]">
        <KpiStrip
          items={[
            {
              value: fmt(data.ablation.find((a) => a.highlighted)?.f1 ?? null, 4),
              label: "In-domain F1 · EC-XGB",
              color: TEAL,
            },
            { value: pct(ecxgb?.flagged, 0), label: "Flagged · RAGTruth", color: ROSE },
            { value: fmt(data.displayEce), label: "Display ECE · RAGTruth QA", color: ACCENT },
            { value: ms(data.latencyP50), label: "Median analysis time", color: AMBER },
          ]}
        />
      </div>
    </SlideFrame>
  );
}

/* ──────────────────────── 20 · Thank you ────────────────────────── */

function ThankYouSlide({ index }: SlideProps) {
  return (
    <div className="relative flex h-full w-full flex-col items-center justify-center overflow-hidden bg-white px-[8cqw] text-center">
      <div
        className="pointer-events-none absolute right-[-14cqw] top-[-22cqh] h-[62cqh] w-[62cqh] rounded-full"
        style={{ background: ACCENT, opacity: 0.07 }}
      />
      <div
        className="pointer-events-none absolute bottom-[-26cqh] left-[-12cqw] h-[50cqh] w-[50cqh] rounded-full"
        style={{ background: TEAL, opacity: 0.06 }}
      />
      <h1 className="relative text-[10.5cqh] font-extrabold leading-none" style={{ color: NEAR_BLACK }}>
        Thank You
      </h1>
      <div className="relative mt-[3cqh] text-[3.3cqh] font-bold" style={{ color: ACCENT }}>
        CSE 4889 - Machine Learning · Section E · Team Phantom Devs
      </div>
      <div className="relative mt-[1cqh] text-[2.6cqh] font-semibold" style={{ color: DEEP_INK }}>
        HaluRISC — Calibrated &amp; Explainable Hallucination Risk
      </div>
      <div className="relative mt-[3.4cqh] flex items-center justify-center gap-[1cqw]">
        <Chip color={TEAL} bg="#ccfbf1">
          Supervisor: Ohidujjaman Tuhin
        </Chip>
        <Chip color={ACCENT} bg="#e0e7ff">
          Team: Phantom Devs
        </Chip>
        <Chip color={AMBER} bg="#fef3c7">
          Section: E
        </Chip>
      </div>
      <div className="relative mt-[3.6cqh] text-[2.4cqh] font-medium" style={{ color: SLATE }}>
        Questions &amp; Discussion Welcome
      </div>
      <div
        className="absolute bottom-[2cqh] inset-x-[4.6cqw] flex items-center justify-end border-t-2 pt-[1.1cqh] text-[1.7cqh] font-bold"
        style={{ borderColor: "#e2e8f0", color: SLATE }}
      >
        <span className="tnum">{index + 1}</span>
      </div>
    </div>
  );
}

/* ─────────────────── 4 · Objective and contributions ─────────────── */

function ObjectiveSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_OBJECTIVE}
      title="Objective and contributions"
      subtitle="Estimate the probability that an LLM answer is hallucinated, and keep that estimate calibrated and explainable under domain shift."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.15fr_1fr] gap-[1.6cqw]">
        <SlideShot
          src="/slides/objective-pipeline.png"
          alt="The question, context, and answer feed 35 features into EC-XGB, and calibration produces a risk score, claim verdicts, and a SHAP explanation"
          caption="From three inputs to a calibrated, explained risk score"
          color={ACCENT}
          ratio="2.5"
        />
        <div className="flex min-h-0 flex-col justify-center gap-[1.2cqh]">
          <Card icon={<Target size="2.2cqh" color="#fff" />} title="What we set out to do" color={ACCENT} fill="#eef2ff">
            <ul>
              <Bullet>Score an answer from its question and evidence, with no model access.</Bullet>
              <Bullet>Report a probability that can be read as a probability.</Bullet>
              <Bullet>Explain every score and test it under domain shift.</Bullet>
            </ul>
          </Card>
          <Card icon={<Lightbulb size="2.2cqh" color="#fff" />} title="Contributions" color={TEAL} fill="#f0fdfa">
            <ul>
              <Bullet>An evidence-consistent feature set of {data.totalFeatures} features.</Bullet>
              <Bullet>Three cumulative changes over the tuned baseline.</Bullet>
              <Bullet>A deployed system with claim-level verification.</Bullet>
            </ul>
          </Card>
        </div>
      </div>
      <div className="mt-[1.2cqh]">
        <KpiStrip
          items={[
            { value: pct(data.shift[0]?.standard.flagged, 1), label: "Standard flags · RAGTruth", color: ROSE },
            { value: pct(data.shift[0]?.ecxgb.flagged, 0), label: "EC-XGB flags · RAGTruth", color: TEAL },
            { value: fmt(data.displayEce), label: "Display ECE after calibration", color: ACCENT },
            { value: ms(data.latencyP50), label: "Median analysis time", color: AMBER },
          ]}
        />
      </div>
    </SlideFrame>
  );
}

/* ─────────────────── 8 · Full architecture flow ──────────────────── */

function FlowSlide({ index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_FLOW}
      title="Full architecture, start to end"
      subtitle="Inputs, feature extraction, the EC-XGB classifier, calibration, outputs, and the deployed evaluation."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="flex min-h-0 flex-1 items-center justify-center">
        <SlideShot
          src="/slides/architecture-flow.svg"
          alt="End-to-end architecture: question, context, and answer feed 35 features, then EC-XGB, Platt calibration, and the risk score, claim verdicts, and SHAP outputs, deployed through Next.js and FastAPI and evaluated on HaluEval, RAGTruth, and FaithBench"
          caption="End-to-end architecture of the deployed system"
          color={ACCENT}
          ratio="16 / 9"
          fill
          background="#ffffff"
        />
      </div>
    </SlideFrame>
  );
}

/* ─────────────────── 12 · Application and deployment ─────────────── */

function ApplicationSlide({ data, index, total }: SlideProps) {
  const judgeName = data.judgeModel === "gpt-5.6-luna" ? "GPT 5.6 Luna" : (data.judgeModel ?? "LLM judge");
  return (
    <SlideFrame
      {...BADGE_APPLICATION}
      title="Applications"
      subtitle="Where a fast, calibrated, explainable hallucination check fits into real workflows."
      accent={TEAL}
      index={index}
      total={total}
    >
      <KpiStrip
        items={[
          { value: fmt(data.displayEce), label: "Display ECE · calibrated", color: ACCENT },
          { value: ms(data.latencyP50), label: "Median analysis", color: AMBER },
          { value: `$${data.costPer1k ?? "—"}`, label: "Cost per 1,000", color: TEAL },
          { value: fmt(data.judgeF1), label: `Judge F1 · ${judgeName}`, color: ROSE },
        ]}
      />
      <div className="mt-[1.3cqh] grid min-h-0 flex-1 grid-cols-[1fr_1.05fr] gap-[1.6cqw]">
        <div className="flex min-h-0 flex-col justify-center">
          <Figure
            src="/slides/application-use-cases.png"
            alt="Three use cases: a chat assistant, document QA with a magnifying glass, and content review with checks and a flag"
            caption="Chat assistants, document QA, and content review"
            height="44cqh"
          />
        </div>
        <div className="flex min-h-0 flex-col justify-center gap-[1.1cqh]">
          <Card icon={<Users size="2.2cqh" color="#fff" />} title="Where it applies" color={ACCENT} fill="#eef2ff">
            <ul>
              <Bullet>Chat assistants that check every answer as it streams.</Bullet>
              <Bullet>Document QA over an indexed corpus or the open web.</Bullet>
              <Bullet>Content review that flags unsupported claims in drafts and summaries.</Bullet>
              <Bullet>Any closed LLM, since no model weights are needed.</Bullet>
            </ul>
          </Card>
          <Card icon={<TrendingUp size="2.2cqh" color="#fff" />} title="What users get" color={AMBER} fill="#fffbeb">
            <ul>
              <Bullet>A calibrated risk score that reads as a probability.</Bullet>
              <Bullet>Per-claim verdicts with the evidence sentence quoted.</Bullet>
              <Bullet>SHAP reasons behind every score, stable under edits.</Bullet>
              <Bullet>Local CPU checks at about ${data.costPer1k ?? "—"} per 1,000 answers.</Bullet>
            </ul>
          </Card>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ------------------------------------------------------------------ */

export const SLIDES: Array<(props: SlideProps) => React.ReactElement> = [
  TitleSlide,
  ProblemSlide,
  MotivationSlide,
  ObjectiveSlide,
  DatasetsSlide,
  ConventionalSlide,
  ProposedSlide,
  FlowSlide,
  ProtocolSlide,
  BaselineResultsSlide,
  EcXgbResultsSlide,
  ApplicationSlide,
  ChatAnalyzeSlide,
  ExplainCompareSlide,
  ConclusionSlide,
  ThankYouSlide,
];
