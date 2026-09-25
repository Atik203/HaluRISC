import {
  AlertTriangle,
  ArrowDown,
  BarChart3,
  Brain,
  Crown,
  Database,
  Gauge,
  GraduationCap,
  Layers,
  Lightbulb,
  ListChecks,
  Monitor,
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
  PipelineStrip,
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
  f1?: number | null;
  auroc?: number | null;
  mcc?: number | null;
  ece?: number | null;
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
  standard: { f1?: number | null; auroc?: number | null; flagged?: number | null; ece?: number | null };
  ecxgb: { f1?: number | null; auroc?: number | null; flagged?: number | null; ece?: number | null };
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
  displayMethod: string;
  calibrationRows: number | null;
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
  modelJudgeF1?: number | null;
  strictAlphaText: string;
  strictFprText: string;
  strictRecallText: string;
  fpNote: string;
}

type SlideProps = { data: SlideData; index: number; total: number };

const BADGE_INTRO = { badge: "01 · Introduction", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_DATA = { badge: "02 · Dataset", badgeBg: "#ccfbf1", badgeColor: TEAL };
const BADGE_METHOD = { badge: "03 · Methodology", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_RESULTS = { badge: "04 · Results", badgeBg: "#fee2e2", badgeColor: ROSE };
const BADGE_UI = { badge: "05 · UI/UX Design", badgeBg: "#e0e7ff", badgeColor: ACCENT };
const BADGE_DEMO = { badge: "06 · Demonstration", badgeBg: "#fef3c7", badgeColor: AMBER };
const BADGE_END = { badge: "07 · Conclusion", badgeBg: "#ccfbf1", badgeColor: TEAL };

/* ───────────────────────────── 1 · Title ─────────────────────────── */

const MEMBERS = [
  { name: "Md. Atikur Rahaman", id: "0112310298", leader: true },
  { name: "Saiful Alam Sabbir", id: "0112310105", leader: false },
  { name: "MD. Miraz Ahamed", id: "0112310524", leader: false },
];

function TitleSlide({ index, total }: SlideProps) {
  return (
    <div className="relative flex h-full w-full flex-col justify-between overflow-hidden bg-white px-[6cqw] pb-[1.6cqh] pt-[3cqh]">
      <div
        className="pointer-events-none absolute right-[-12cqw] top-[-20cqh] h-[56cqh] w-[56cqh] rounded-full"
        style={{ background: ACCENT, opacity: 0.07 }}
      />
      <div
        className="pointer-events-none absolute bottom-[-24cqh] left-[-10cqw] h-[44cqh] w-[44cqh] rounded-full"
        style={{ background: TEAL, opacity: 0.06 }}
      />
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
          Estimation in Black-Box Language Model Answers
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
                {m.leader && <span className="ml-[0.6cqw] font-extrabold">· Leader</span>}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div
        className="relative flex items-center justify-between border-t-2 pt-[0.9cqh] text-[1.7cqh] font-bold"
        style={{ borderColor: "#e2e8f0", color: SLATE }}
      >
        <span>Video presentation · 8 minutes</span>
        <span className="tnum">
          {index + 1} / {total}
        </span>
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
        <Card icon={<AlertTriangle size="2.4cqh" color="#fff" />} title="A real case" color={ROSE} fill="#fff1f2">
          <div className="space-y-[1.2cqh] text-[2.35cqh] leading-snug" style={{ color: NEAR_BLACK }}>
            <p>
              <b>Context:</b> the Arctic melt season lengthened at <b>5 days per decade</b>.
            </p>
            <p>
              <b>Answer:</b> &ldquo;The melt season has lengthened by{" "}
              <b style={{ color: ROSE }}>10 days</b> per decade.&rdquo;
            </p>
            <p className="rounded-lg px-[1.2cqw] py-[1cqh]" style={{ background: "#ffffff" }}>
              Fluent, confident, and contradicted by the evidence.
            </p>
          </div>
        </Card>
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
          className="text-[1.9cqh] font-extrabold uppercase tracking-wide"
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
      {...BADGE_INTRO}
      title="Three gaps this project closes"
      subtitle="Calibration, explanation reliability, and cross-domain behaviour, tested in one study."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-3 gap-[1.6cqw]">
        {gaps.map((gap) => (
          <Card key={gap.title} icon={gap.icon} title={gap.title} color={gap.color} fill={gap.fill}>
            <ul>
              {gap.lines.map((line) => (
                <Bullet key={line}>{line}</Bullet>
              ))}
            </ul>
          </Card>
        ))}
      </div>
      <div className="mt-[1.4cqh]">
        <div
          className="mb-[0.7cqh] text-[1.9cqh] font-extrabold uppercase tracking-wide"
          style={{ color: SLATE }}
        >
          What the study delivers
        </div>
        <KpiStrip
          items={[
            {
              value: `${pct(standard?.flagged, 0)} → ${pct(ecxgb?.flagged, 0)}`,
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

/* ────────────────────── 4 · Dataset (in-domain) ─────────────────── */

function DatasetInDomainSlide({ index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_DATA}
      title="Training data: HaluEval QA"
      subtitle="Paired correct and hallucinated answers for the same question, so the model learns evidence consistency."
      accent={TEAL}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.05fr_1fr] gap-[1.8cqw]">
        <Card icon={<Database size="2.4cqh" color="#fff" />} title="What is inside" color={TEAL} fill="#f0fdfa">
          <ul>
            <Bullet>
              <b>10,000 questions</b>, each with a knowledge passage and a pair of answers.
            </Bullet>
            <Bullet>
              <b>20,000 labeled rows</b> after pairing, binary target: hallucinated or grounded.
            </Bullet>
            <Bullet>
              Answers are often terse: <b>style itself carries signal</b>, which later motivates the
              length features and the debiased variant.
            </Bullet>
            <Bullet>Benchmark caveat: most rows are built with a sampling-then-filtering procedure.</Bullet>
          </ul>
        </Card>
        <div className="flex min-h-0 flex-col justify-center gap-[1.5cqh]">
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
            className="rounded-lg px-[1.4cqw] py-[1cqh] text-[2.05cqh] font-bold"
            style={{ background: "#ccfbf1", color: TEAL }}
          >
            Grouped split verified leakage-free: both answers of a question stay in one partition.
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ──────────────────── 5 · Dataset (external + features) ─────────── */

function DatasetExternalSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_DATA}
      title="External corpora and the feature vector"
      subtitle="Natural responses test transfer; claim-level aggregates connect the model to the verification layer."
      accent={TEAL}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.05fr_1fr] gap-[1.8cqw]">
        <div className="flex min-h-0 flex-col justify-center gap-[1.4cqh]">
          <SlideTable
            head={["Corpus", "Rows", "Role"]}
            headerBg="#ccfbf1"
            color={TEAL}
            widths={["32%", "20%", "48%"]}
            rows={[
              ["RAGTruth (all)", "17,790", "Zero-shot transfer"],
              ["RAGTruth QA", "5,934", "5,034 calibration + 900 test"],
              ["FaithBench", "750", "Summarization stress test"],
            ]}
          />
          <div
            className="rounded-lg px-[1.4cqw] py-[1cqh] text-[2.05cqh] font-bold leading-snug"
            style={{ background: "#fef3c7", color: AMBER }}
          >
            Zero-shot rows never enter training. RAGTruth QA stays fully held out, so its result is a
            zero-shot task for the deployed model.
          </div>
        </div>
        <Card
          icon={<Layers size="2.4cqh" color="#fff" />}
          title={`Feature vector · ${data.totalFeatures} features`}
          color={ACCENT}
          fill="#eef2ff"
        >
          <div className="space-y-[1.2cqh]">
            <StackedBar
              height="4cqh"
              segments={[
                { label: `${data.baseFeatures} base`, value: data.baseFeatures, color: ACCENT },
                { label: `${data.claimFeatures} claim`, value: data.claimFeatures, color: TEAL },
                { label: "1", value: 1, color: AMBER },
              ]}
            />
            <div className="space-y-[0.9cqh] text-[2.25cqh] font-semibold" style={{ color: NEAR_BLACK }}>
              <div>
                <b>{data.baseFeatures} base:</b> lexical, entity, NLI, numeric, hedging, semantic, length.
              </div>
              <div>
                <b>{data.claimFeatures} claim:</b> per-claim support, contradiction, and coverage ratios.
              </div>
              <div>
                <b>1 source:</b> natural-response indicator used by the multi-source variant.
              </div>
            </div>
            <p className="text-[2cqh] font-medium leading-snug" style={{ color: SLATE }}>
              Claim features use atomic clauses (conjunctions split) scored against the four best-matching
              context sentences.
            </p>
          </div>
        </Card>
      </div>
    </SlideFrame>
  );
}

/* ─────────────────────────── 6 · Pipeline ───────────────────────── */

function PipelineBox({
  title,
  sub,
  color,
  bg,
  wide = false,
}: {
  title: string;
  sub?: string;
  color: string;
  bg: string;
  wide?: boolean;
}) {
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border-2 px-[1.4cqw] py-[1.2cqh] text-center ${wide ? "w-full" : ""}`}
      style={{ borderColor: color, background: bg }}
    >
      <div className="text-[2.35cqh] font-extrabold" style={{ color }}>
        {title}
      </div>
      {sub && (
        <div className="mt-[0.35cqh] text-[1.95cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
          {sub}
        </div>
      )}
    </div>
  );
}

function PipelineSlide({ data, index, total }: SlideProps) {
  const groupNames = Object.keys(data.featureGroups);
  return (
    <SlideFrame
      {...BADGE_METHOD}
      title="End-to-end system"
      subtitle="One pass from question, context, and answer to a calibrated score with per-claim verdicts."
      accent={AMBER}
      index={index}
      total={total}
    >
      <div className="flex min-h-0 flex-1 flex-col justify-center gap-[0.9cqh]">
        <div className="grid grid-cols-3 gap-[1.2cqw]">
          <PipelineBox title="Question" color={ACCENT} bg="#eef2ff" />
          <PipelineBox title="Context / Evidence" color={ACCENT} bg="#eef2ff" />
          <PipelineBox title="Answer (black-box LLM)" color={ACCENT} bg="#eef2ff" />
        </div>
        <div className="flex justify-center">
          <ArrowDown size="2.6cqh" style={{ color: AMBER }} />
        </div>
        <PipelineBox
          wide
          title={`Feature extraction · ${data.totalFeatures} features`}
          sub={groupNames.join(" · ")}
          color={AMBER}
          bg="#fffbeb"
        />
        <div className="flex justify-center">
          <ArrowDown size="2.6cqh" style={{ color: TEAL }} />
        </div>
        <div className="grid grid-cols-[1.3fr_1fr] gap-[1.2cqw]">
          <PipelineBox
            title="EC-XGB · Evidence-Consistent XGBoost"
            sub="monotone constraints · claim aggregates · multi-source training"
            color={TEAL}
            bg="#f0fdfa"
          />
          <PipelineBox
            title="Display calibrator"
            sub={`${data.displayMethod} on natural RAGTruth QA rows`}
            color={ACCENT}
            bg="#eef2ff"
          />
        </div>
        <div className="flex justify-center">
          <ArrowDown size="2.6cqh" style={{ color: ROSE }} />
        </div>
        <div className="grid grid-cols-3 gap-[1.2cqw]">
          <PipelineBox title="Calibrated risk score" color={ROSE} bg="#fff1f2" />
          <PipelineBox title="Per-claim verdicts" color={ROSE} bg="#fff1f2" />
          <PipelineBox title="SHAP explanations" color={ROSE} bg="#fff1f2" />
        </div>
      </div>
      <div className="mt-[1.3cqh]">
        <PipelineStrip
          steps={["Paste or stream input", "Score + verdicts", "Inspect evidence", "Compare models", "Deploy"]}
          color={ACCENT}
          bg="#eef2ff"
        />
      </div>
    </SlideFrame>
  );
}

/* ────────────────────── 7 · Feature engineering ─────────────────── */

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

function FeaturesSlide({ data, index, total }: SlideProps) {
  const groups = Object.entries(data.featureGroups);
  return (
    <SlideFrame
      {...BADGE_METHOD}
      title="Feature engineering"
      subtitle={`${data.baseFeatures} base features in seven groups plus ${data.claimFeatures} claim-level NLI aggregates.`}
      accent={AMBER}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.15fr_1fr] gap-[1.6cqw]">
        <SlideTable
          compact
          head={["Group", "Features", "Count"]}
          headerBg="#fef3c7"
          color={AMBER}
          widths={["30%", "54%", "16%"]}
          highlightRow={groups.length - 1}
          rows={groups.map(([key, cols]) => [
            GROUP_LABELS[key] ?? key,
            cols.join(", "),
            String(cols.length),
          ])}
        />
        <div className="flex min-h-0 flex-col gap-[1.2cqh]">
          <Card
            icon={<Sigma size="2.4cqh" color="#fff" />}
            title="Why claim features"
            color={TEAL}
            fill="#f0fdfa"
          >
            <ul>
              <Bullet>Max and mean contradiction, contradicted-claim ratio.</Bullet>
              <Bullet>Supported and unsupported claim ratios.</Bullet>
              <Bullet>Min and mean entailment across claims.</Bullet>
              <Bullet>Verdicts are support-first with a relevance gate.</Bullet>
            </ul>
          </Card>
          <div
            className="rounded-xl border-2 px-[1.6cqw] py-[1.3cqh]"
            style={{ borderColor: ACCENT, background: "#eef2ff" }}
          >
            <div className="text-[2cqh] font-extrabold uppercase tracking-wide" style={{ color: ACCENT }}>
              Atomic clause splitting
            </div>
            <div className="mt-[0.8cqh] text-[2.1cqh] font-semibold leading-snug" style={{ color: NEAR_BLACK }}>
              &ldquo;The change is dominated by a later freezeup, <b>and</b> the region is at its
              warmest&hellip;&rdquo;
            </div>
            <div className="mt-[0.8cqh] flex items-center gap-[0.8cqw]">
              <Chip color={TEAL} bg="#ccfbf1">
                claim 1 · supported
              </Chip>
              <Chip color={TEAL} bg="#ccfbf1">
                claim 2 · supported
              </Chip>
            </div>
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ───────────────────────── 8 · Models ──────────────────────────── */

function ModelsSlide({ data, index, total }: SlideProps) {
  const baselines = [
    { title: "Overlap heuristic", color: SLATE, lines: ["Risk = 1 − lexical overlap", "Threshold tuned on validation"] },
    { title: "Logistic regression", color: ACCENT, lines: ["Standardized features", "Linear decision boundary"] },
    { title: "Random forest", color: TEAL, lines: ["300 trees", "Strong tabular baseline"] },
    { title: "XGBoost", color: AMBER, lines: ["30-iteration randomized search", "Grouped 5-fold CV"] },
  ];
  return (
    <SlideFrame
      {...BADGE_METHOD}
      title="Models compared, and the EC-XGB variant"
      subtitle="Four standard baselines, then three changes that define the deployed model."
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
      <div className="mt-[1.6cqh] grid min-h-0 flex-1 grid-cols-[1.4fr_1fr] gap-[1.6cqw]">
        <div
          className="flex h-full flex-col rounded-xl border-[3px] px-[2.2cqw] py-[1.6cqh]"
          style={{ borderColor: ACCENT, background: "#eef2ff" }}
        >
          <div className="flex items-center gap-[1cqw]">
            <Brain size="3.2cqh" style={{ color: ACCENT }} />
            <span className="text-[3.1cqh] font-extrabold" style={{ color: ACCENT }}>
              EC-XGB · Evidence-Consistent XGBoost
            </span>
          </div>
          <div className="mt-[1.3cqh] grid flex-1 grid-cols-3 gap-[1.2cqw]">
            {["m1", "m2", "m3"].map((key) => (
              <div
                key={key}
                className="flex flex-col justify-center rounded-lg border-2 bg-white px-[1.4cqw] py-[1.1cqh]"
                style={{ borderColor: "#c7d2fe" }}
              >
                <div className="text-[2.1cqh] font-extrabold" style={{ color: ACCENT }}>
                  {key.toUpperCase()}
                </div>
                <div className="mt-[0.45cqh] text-[1.95cqh] font-medium leading-snug" style={{ color: DEEP_INK }}>
                  {data.componentMap[key] ?? "—"}
                </div>
              </div>
            ))}
          </div>
        </div>
        <Card icon={<ShieldCheck size="2.4cqh" color="#fff" />} title="Monotone evidence rules" color={TEAL} fill="#f0fdfa">
          <ul>
            <Bullet>Risk cannot fall when contradiction rises.</Bullet>
            <Bullet>Risk cannot climb when entailment, overlap, or cosine rises.</Bullet>
            <Bullet>Strict mode caps the validation false-positive budget at {data.strictAlphaText}.</Bullet>
            <Bullet>Claim features and full confidence aggregates are both visible to the learner.</Bullet>
          </ul>
        </Card>
      </div>
    </SlideFrame>
  );
}

/* ──────────────────────── 9 · Training protocol ─────────────────── */

function ProtocolSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_METHOD}
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
            Grouped 5-fold CV.
            <br />
            {data.nIter ?? "—"}-iteration randomized search over depth, learning rate, trees, and subsampling.
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
        <PipelineStrip
          steps={["Features (35)", "Grouped CV", "Train + early stop", "Calibrate on validation", "Test + shift metrics"]}
          color={TEAL}
          bg="#f0fdfa"
        />
        <div className="mt-[1cqh] flex items-center gap-[0.8cqw]">
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

/* ───────────────────── 10 · Baseline comparison ─────────────────── */

function BaselineResultsSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_RESULTS}
      title="Baseline comparison on HaluEval QA"
      subtitle="Test set, mean over seeds 42/123/456. ECE measures how reliable each model's risk score is."
      accent={ROSE}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.2fr_1fr] gap-[1.8cqw]">
        <div className="flex min-h-0 flex-col justify-center gap-[1.2cqh]">
          <SlideTable
            head={["Model", "F1", "AUROC", "MCC", "ECE"]}
            headerBg="#fee2e2"
            color={ROSE}
            widths={["40%", "15%", "15%", "15%", "15%"]}
            highlightRow={data.baselines.findIndex((b) => b.highlighted)}
            rows={data.baselines.map((b) => [b.label, fmt(b.f1), fmt(b.auroc), fmt(b.mcc), fmt(b.ece, 4)])}
          />
          <div
            className="rounded-lg px-[1.4cqw] py-[1cqh] text-[2.05cqh] font-bold leading-snug"
            style={{ background: "#fee2e2", color: ROSE }}
          >
            {data.fpNote || "XGBoost shows the fewest false positives at the same threshold."}
          </div>
        </div>
        <div className="flex min-h-0 flex-col items-center justify-center gap-[1.1cqh]">
          <Figure
            src="/api/figures/fig_roc_pr.png"
            alt="ROC and precision-recall curves on the HaluEval test set"
            height="30cqh"
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

/* ───────────────────── 11 · EC-XGB results ──────────────────────── */

function EcXgbResultsSlide({ data, index, total }: SlideProps) {
  const standard = data.shift[0]?.standard;
  const ecxgb = data.shift[0]?.ecxgb;
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
            value: `${pct(standard?.flagged, 0)} → ${pct(ecxgb?.flagged, 0)}`,
            label: "Flagged as risky · RAGTruth",
            color: ROSE,
          },
          { value: `${fmt(standard?.auroc)} → ${fmt(ecxgb?.auroc)}`, label: "AUROC · RAGTruth", color: TEAL },
          { value: `${fmt(standard?.ece)} → ${fmt(ecxgb?.ece)}`, label: "ECE · RAGTruth", color: ACCENT },
        ]}
      />
      <div className="mt-[1.5cqh] grid min-h-0 flex-1 grid-cols-2 gap-[1.6cqw]">
        <div className="flex min-h-0 flex-col gap-[0.8cqh]">
          <div className="text-[2cqh] font-extrabold uppercase tracking-wide" style={{ color: SLATE }}>
            Component ablation (in-domain)
          </div>
          <SlideTable
            head={["Variant", "F1", "AUROC", "ECE"]}
            headerBg="#e0e7ff"
            color={ACCENT}
            widths={["46%", "18%", "18%", "18%"]}
            highlightRow={data.ablation.findIndex((a) => a.highlighted)}
            rows={data.ablation.map((a) => [a.label, fmt(a.f1, 4), fmt(a.auroc), fmt(a.ece, 4)])}
          />
          <div
            className="rounded-lg px-[1.3cqw] py-[0.9cqh] text-[1.95cqh] font-bold leading-snug"
            style={{ background: "#eef2ff", color: ACCENT }}
          >
            McNemar m0 vs m3 is not significant in-domain, so the value is robustness, not extra
            benchmark points.
          </div>
        </div>
        <div className="flex min-h-0 flex-col gap-[0.8cqh]">
          <div className="text-[2cqh] font-extrabold uppercase tracking-wide" style={{ color: SLATE }}>
            Standard vs EC-XGB on shifted corpora
          </div>
          <SlideTable
            head={["Corpus", "Model", "F1", "AUROC", "Flagged"]}
            headerBg="#ccfbf1"
            color={TEAL}
            widths={["30%", "22%", "16%", "16%", "16%"]}
            rows={data.shift.flatMap((row) => [
              [row.datasetLabel, "Standard", fmt(row.standard.f1), fmt(row.standard.auroc), pct(row.standard.flagged)],
              [row.datasetLabel, "EC-XGB", fmt(row.ecxgb.f1), fmt(row.ecxgb.auroc), pct(row.ecxgb.flagged)],
            ])}
          />
          <div
            className="rounded-lg px-[1.3cqw] py-[0.9cqh] text-[1.95cqh] font-bold leading-snug"
            style={{ background: "#f0fdfa", color: TEAL }}
          >
            On unseen corpora EC-XGB trades recall for precision and raises AUROC while the flag rate
            collapses.
          </div>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ──────────────── 12 · Calibration, trust, and speed ────────────── */

function TrustResultsSlide({ data, index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_RESULTS}
      title="Calibrated, explainable, and fast"
      subtitle="The deployed score is trustworthy under shift, the explanations are stable, and inference stays light."
      accent={TEAL}
      index={index}
      total={total}
    >
      <div className="grid grid-cols-3 gap-[1.5cqw]">
        <Card icon={<Gauge size="2.2cqh" color="#fff" />} title="Calibration" color={ACCENT} fill="#eef2ff">
          <div className="tnum text-[6.2cqh] font-extrabold leading-none" style={{ color: ACCENT }}>
            {fmt(data.displayRawEce)} → {fmt(data.displayEce)}
          </div>
          <div className="mt-[0.8cqh] text-[2.15cqh] font-bold" style={{ color: NEAR_BLACK }}>
            ECE on RAGTruth QA · raw to {data.displayMethod}
          </div>
          <div className="mt-[0.5cqh] text-[1.95cqh] font-medium leading-snug" style={{ color: SLATE }}>
            Brier {fmt(data.displayRawBrier)} → {fmt(data.displayBrier)} · fitted on{" "}
            {data.calibrationRows?.toLocaleString() ?? "—"} calibration rows.
          </div>
        </Card>
        <Card icon={<BarChart3 size="2.2cqh" color="#fff" />} title="Explanation reliability" color={TEAL} fill="#f0fdfa">
          <ul>
            <Bullet>SHAP vs permutation: Kendall τ = {fmt(data.kendall, 2)}.</Bullet>
            <Bullet>Top-5 SHAP set stable across 1,000 bootstrap resamples (Jaccard {fmt(data.jaccard, 2)}).</Bullet>
            <Bullet>Entity edits move the score (Δ {fmt(data.entityDelta, 3)}), irrelevant inserts do not (Δ {fmt(data.irrelevantDelta, 3)}).</Bullet>
          </ul>
        </Card>
        <Card icon={<TrendingUp size="2.2cqh" color="#fff" />} title="Efficiency & cost" color={AMBER} fill="#fffbeb">
          <ul>
            <Bullet>One analysis in {ms(data.latencyP50)} at the median.</Bullet>
            <Bullet>Model artifact {data.artifactMb ?? "—"} MB.</Bullet>
            <Bullet>About ${data.costPer1k ?? "—"} per 1,000 predictions.</Bullet>
            <Bullet>
              LLM judge baseline F1 {fmt(data.judgeF1)} vs {fmt(data.modelJudgeF1)} here, at roughly 100×
              the cost.
            </Bullet>
          </ul>
        </Card>
      </div>
      <div className="mt-[1.4cqh] min-h-0 flex-1">
        <Figure
          src="/api/figures/b4/reliability_diagrams.png"
          alt="Reliability diagrams for HaluEval, RAGTruth QA, and FaithBench"
          caption="Reliability diagrams: in-domain, target-domain, and stress-test calibration"
          height="30cqh"
        />
      </div>
    </SlideFrame>
  );
}

/* ───────────────────────── 13 · UI: chat ───────────────────────── */

function ChatUiSlide({ index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_UI}
      title="Chat and Analyze: evidence-first interface"
      subtitle="The chat card leads with per-claim verdicts; Analyze shows the gauge, thresholds, SHAP bars, and the model comparison."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-3 gap-[1.5cqw]">
        <SlideShot
          src="/slides/chat-grounded.png"
          alt="Chat page with streaming answer and automatic risk card"
          caption="Chat · automatic risk card"
          color={ACCENT}
          position="center 35%"
        />
        <SlideShot
          src="/slides/analyze-risk.png"
          alt="Analyze page with calibrated gauge, thresholds and SHAP"
          caption="Analyze · gauge, thresholds, SHAP"
          color={TEAL}
          position="right top"
        />
        <SlideShot
          src="/slides/analyze-compare.png"
          alt="Model comparison card with EC-XGB beside the standard baselines"
          caption="Analyze · model comparison card"
          color={AMBER}
          position="right top"
        />
      </div>
      <div className="mt-[1.2cqh] flex flex-wrap items-center gap-[0.9cqw]">
        <Chip color={ACCENT} bg="#e0e7ff">
          Dark theme · high contrast
        </Chip>
        <Chip color={TEAL} bg="#ccfbf1">
          Claim verdicts with evidence quotes
        </Chip>
        <Chip color={AMBER} bg="#fef3c7">
          Floating composer: upload, evidence, web search
        </Chip>
        <Chip color={ROSE} bg="#fee2e2">
          All models scored side by side
        </Chip>
      </div>
    </SlideFrame>
  );
}

/* ────────────────────── 14 · UI: dashboard ─────────────────────── */

function DashboardUiSlide({ index, total }: SlideProps) {
  return (
    <SlideFrame
      {...BADGE_UI}
      title="Dashboard, demo, and usability"
      subtitle="Every reported number renders from the artifacts, so the interface and the paper never disagree."
      accent={ACCENT}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-[1.35fr_1fr] gap-[1.6cqw]">
        <SlideShot
          src="/slides/dashboard-robustness.png"
          alt="Robustness tab with the EC-XGB shift and ablation panels"
          caption="Experiment dashboard · EC-XGB shift and ablation panels"
          color={ACCENT}
          position="left top"
        />
        <div className="flex min-h-0 flex-col gap-[1.2cqh]">
          <div className="min-h-0 flex-1">
            <SlideShot
              src="/slides/demo-ecxgb.png"
              alt="Offline presenter demo with the EC-XGB shift table"
              caption="Offline presenter demo · no API key needed"
              color={TEAL}
              position="center top"
            />
          </div>
          <Card icon={<Monitor size="2.2cqh" color="#fff" />} title="Usability checks" color={AMBER} fill="#fffbeb">
            <ul>
              <Bullet>Thresholds and bands are printed on the gauge.</Bullet>
              <Bullet>Result tables carry plain-language notes.</Bullet>
              <Bullet>Projector mode scales the interface.</Bullet>
              <Bullet>Mobile keeps chat, sidebar, and composer reachable.</Bullet>
            </ul>
          </Card>
        </div>
      </div>
    </SlideFrame>
  );
}

/* ─────────────────────────── 15 · Demo ──────────────────────────── */

function DemoSlide({ index, total }: SlideProps) {
  const steps = [
    {
      color: TEAL,
      bg: "#f0fdfa",
      title: "1 · Grounded answer",
      lines: [
        "Q: Who discovered penicillin?",
        "C: Penicillin was discovered by Alexander Fleming in 1928.",
        "A: Penicillin was discovered by Alexander Fleming in 1928.",
      ],
      expected: "Expect low risk · all claims supported",
    },
    {
      color: ROSE,
      bg: "#fff1f2",
      title: "2 · Hallucinated number",
      lines: [
        "Q: How many days per decade did the melt season lengthen?",
        "C: the rate is 5 days per decade.",
        "A: the melt season lengthened by 10 days per decade.",
      ],
      expected: "Expect high risk · contradicted with evidence quote",
    },
    {
      color: ACCENT,
      bg: "#eef2ff",
      title: "3 · Compare and inspect",
      lines: [
        "Paste two answers in Analyze mode.",
        "Open the model comparison: EC-XGB, standard XGBoost, logistic regression, random forest, heuristic.",
        "Expand claim verdicts and SHAP bars.",
      ],
      expected: "Expect EC-XGB to separate the pair",
    },
  ];
  return (
    <SlideFrame
      {...BADGE_DEMO}
      title="Live walkthrough"
      subtitle="Screen recording: paste an answer, score it, inspect claims and explanations, then compare models."
      accent={AMBER}
      index={index}
      total={total}
    >
      <div className="grid min-h-0 flex-1 grid-cols-3 gap-[1.5cqw]">
        {steps.map((step) => (
          <div
            key={step.title}
            className="flex h-full flex-col rounded-xl border-2 px-[1.8cqw] py-[1.5cqh]"
            style={{ borderColor: step.color, background: step.bg }}
          >
            <div className="text-[2.5cqh] font-extrabold" style={{ color: step.color }}>
              {step.title}
            </div>
            <div className="mt-[1cqh] flex-1 space-y-[0.7cqh]">
              {step.lines.map((line) => (
                <div key={line} className="text-[2cqh] font-medium leading-snug" style={{ color: NEAR_BLACK }}>
                  {line}
                </div>
              ))}
            </div>
            <div
              className="mt-[1cqh] rounded-lg bg-white px-[1.2cqw] py-[0.8cqh] text-[2cqh] font-bold"
              style={{ color: step.color }}
            >
              {step.expected}
            </div>
          </div>
        ))}
      </div>
      <div className="mt-[1.2cqh] flex flex-wrap items-center gap-[0.9cqw]">
        <Chip color={AMBER} bg="#fef3c7">
          ~90 seconds of screen recording
        </Chip>
        <Chip color={ACCENT} bg="#e0e7ff">
          Preload the page before recording
        </Chip>
        <Chip color={TEAL} bg="#ccfbf1">
          Show the evidence quote, not just the score
        </Chip>
      </div>
    </SlideFrame>
  );
}

/* ──────────────────────── 16 · Conclusion ───────────────────────── */

function ConclusionSlide({ data, index, total }: SlideProps) {
  const ecxgb = data.shift[0]?.ecxgb;
  return (
    <SlideFrame
      {...BADGE_END}
      title="What we found, and what remains open"
      subtitle="A light, calibrated, explainable detector that holds up in-domain and flags far fewer natural answers."
      accent={TEAL}
      index={index}
      total={total}
      footerLeft={`Model ${data.modelVersion} · features ${data.featureVersion}`}
    >
      <div className="grid min-h-0 flex-1 grid-cols-3 gap-[1.6cqw]">
        <Card icon={<ShieldCheck size="2.2cqh" color="#fff" />} title="Findings" color={TEAL} fill="#f0fdfa">
          <ul>
            <Bullet>
              In-domain F1 <b>{fmt(data.ablation.find((a) => a.highlighted)?.f1 ?? null, 4)}</b> with
              calibration intact.
            </Bullet>
            <Bullet>
              Shift over-flagging falls from {pct(data.shift[0]?.standard.flagged, 0)} to{" "}
              {pct(ecxgb?.flagged, 0)}.
            </Bullet>
            <Bullet>
              Display ECE drops from {fmt(data.displayRawEce)} to {fmt(data.displayEce)}.
            </Bullet>
            <Bullet>Every number ships with tests, ablations, and a frozen manifest.</Bullet>
          </ul>
        </Card>
        <Card icon={<AlertTriangle size="2.2cqh" color="#fff" />} title="Limitations" color={AMBER} fill="#fffbeb">
          <ul>
            <Bullet>English-only models and features.</Bullet>
            <Bullet>Held-out RAGTruth QA improves mainly in calibration, not F1.</Bullet>
            <Bullet>On FaithBench the model becomes conservative and trades F1 for a much lower flag rate.</Bullet>
            <Bullet>Multi-source gains include task types seen in training.</Bullet>
          </ul>
        </Card>
        <Card icon={<Wrench size="2.2cqh" color="#fff" />} title="Future work" color={ACCENT} fill="#eef2ff">
          <ul>
            <Bullet>Multicalibration under stronger shift.</Bullet>
            <Bullet>Per-domain threshold transfer.</Bullet>
            <Bullet>Neural and verifier baselines beside EC-XGB.</Bullet>
            <Bullet>More languages and more evidence sources.</Bullet>
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

/* ──────────────────────── 17 · Thank you ────────────────────────── */

function ThankYouSlide({ index, total }: SlideProps) {
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
        className="absolute bottom-[1.6cqh] inset-x-[4.2cqw] flex items-center justify-between border-t-2 pt-[0.8cqh] text-[1.7cqh] font-bold"
        style={{ borderColor: "#e2e8f0", color: SLATE }}
      >
        <span>CSE 4889 - Machine Learning · Section E · Team Phantom Devs</span>
        <span className="tnum">
          {index + 1} / {total}
        </span>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */

export const SLIDES: Array<(props: SlideProps) => React.ReactElement> = [
  TitleSlide,
  ProblemSlide,
  MotivationSlide,
  DatasetInDomainSlide,
  DatasetExternalSlide,
  PipelineSlide,
  FeaturesSlide,
  ModelsSlide,
  ProtocolSlide,
  BaselineResultsSlide,
  EcXgbResultsSlide,
  TrustResultsSlide,
  ChatUiSlide,
  DashboardUiSlide,
  DemoSlide,
  ConclusionSlide,
  ThankYouSlide,
];
