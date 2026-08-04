import { openai } from "@ai-sdk/openai";
import { isStepCount, streamText } from "ai";
import { z } from "zod";

export const runtime = "nodejs";

// Static system prompt (prompt-caching rule: no dynamic variables inside).
const SYSTEM_PROMPT = `You are HaluRISC, an expert AI hallucination risk analyst.
You explain predictions produced by a calibrated XGBoost model trained on evidence-consistency
features (length, lexical overlap, entity overlap, NLI entailment/contradiction, numeric
consistency, hedging, semantic similarity).

When a user asks you to check an answer for hallucination, you MUST:
1. Call the analyze_hallucination tool with the question, context, and answer.
2. Present the risk score clearly (calibrated probability, 0-100%).
3. Explain WHY the answer is risky or safe using ONLY the SHAP feature contributions returned by the tool.
4. Never invent risk scores, feature values, or explanations. If the tool returns an error,
   say the ML backend is unavailable instead of fabricating numbers.
5. Ask for context/evidence if the user provides only an answer: risk prediction needs a
   question, a reference context, and the answer to be meaningful.`;

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const result = streamText({
      model: openai(process.env.OPENAI_MODEL || "gpt-5.6-luna"),
      system: SYSTEM_PROMPT,
      messages,
      tools: {
        analyze_hallucination: {
          description:
            "Analyze an answer for hallucination risk using the HaluRISC ML model (calls FastAPI /predict + /explain)",
          inputSchema: z.object({
            question: z.string().describe("The question that was asked"),
            context: z.string().describe("The reference context or evidence"),
            answer: z.string().describe("The answer to analyze for hallucination"),
          }),
          execute: async ({ question, context, answer }) => {
            const backendUrl =
              process.env.NEXT_PUBLIC_ML_API_URL || "http://127.0.0.1:8000";

            const [predRes, expRes] = await Promise.all([
              fetch(`${backendUrl}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, context, answer }),
              }),
              fetch(`${backendUrl}/explain`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question, context, answer }),
              }),
            ]);

            const prediction = predRes.ok
              ? ((await predRes.json()) as { error?: string; calibrated_score?: number; label?: string; latency_ms?: number })
              : { error: `ML backend error: ${predRes.status} ${await predRes.text()}` };
            const explanation = expRes.ok ? ((await expRes.json()) as { top_features?: unknown; base_value?: unknown } | null) : null;

            return { prediction, explanation };
          },
        },
      },
      stopWhen: isStepCount(3),
    });

    return result.toUIMessageStreamResponse();
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }
}
