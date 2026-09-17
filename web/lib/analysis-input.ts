/**
 * B7.5 Tier 1 — auto-analysis input assembly.
 *
 * Pure helpers (no React) so they stay unit-testable. Mirrors the API limits:
 *   question <= 5_000, context <= 20_000, answer <= 20_000 chars.
 */

export const QUESTION_MAX = 5_000;
export const CONTEXT_MAX = 20_000;
export const ANSWER_MAX = 20_000;
const CONVERSATION_TURNS = 3;

export interface Turn {
  role: "user" | "assistant";
  text: string;
}

export interface AnalysisInput {
  question: string;
  context: string;
  answer: string;
  /** Which grounding produced the context (shown on the card). */
  grounding: "evidence" | "conversation";
}

export function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max - 1)}…`;
}

/** Last user message text becomes the question; skip empty. */
export function lastUserText(messages: Turn[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") {
      const t = messages[i].text.trim();
      return t || null;
    }
  }
  return null;
}

/** Build the conversation-window context: up to N turns before the answer. */
export function conversationContext(messages: Turn[], answerIndex: number): string {
  const start = Math.max(0, answerIndex - CONVERSATION_TURNS * 2);
  return messages
    .slice(start, answerIndex)
    .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.text}`)
    .join("\n\n");
}

/**
 * Compose the analysis inputs for the given assistant message.
 * Returns null when there is nothing analyzable (no text, no question, or
 * answer/context over the API limits).
 */
export function buildAnalysisInput(
  messages: Turn[],
  answerIndex: number,
  sessionContext: string | null,
): AnalysisInput | null {
  const answer = (messages[answerIndex]?.text ?? "").trim();
  if (!answer) return null;
  if (answer.length > ANSWER_MAX) return null;

  const question = lastUserText(messages.slice(0, answerIndex));
  if (!question) return null;

  const evidence = (sessionContext ?? "").trim();
  if (evidence) {
    return {
      question: truncate(question, QUESTION_MAX),
      context: truncate(evidence, CONTEXT_MAX),
      answer,
      grounding: "evidence",
    };
  }
  const conv = conversationContext(messages, answerIndex).trim();
  if (!conv) return null;
  return {
    question: truncate(question, QUESTION_MAX),
    context: truncate(conv, CONTEXT_MAX),
    answer,
    grounding: "conversation",
  };
}
