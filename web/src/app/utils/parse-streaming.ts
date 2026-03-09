import { Relate } from "@/app/interfaces/relate";
import { Source } from "@/app/interfaces/source";
import { fetchStream } from "@/app/utils/fetch-stream";

const LLM_SPLIT = "__LLM_RESPONSE__";
const RELATED_SPLIT = "__RELATED_QUESTIONS__";

export const parseStreaming = async (
  controller: AbortController,
  query: string,
  search_uuid: string,
  onSources: (value: Source[]) => void,
  onMarkdown: (value: string) => void,
  onRelates: (value: Relate[]) => void,
  onError?: (status: number) => void,
) => {
  const decoder = new TextDecoder();
  let buffer = "";
  let sourcesEmitted = false;
  let started = false;
  let markdownBuffer = "";
  let relatesBuffer = "";

  const response = await fetch(`/query`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "*/*",
    },
    signal: controller.signal,
    body: JSON.stringify({
      query,
      search_uuid,
      generate_related_questions: true, // match your backend default
    }),
  });

  if (response.status !== 200) {
    onError?.(response.status);
    return;
  }

  fetchStream(
    response,
    (chunk) => {
      buffer += decoder.decode(chunk, { stream: true });

      // Keep processing until we find the LLM response start
      if (!started) {
        const markerIndex = buffer.indexOf(LLM_SPLIT);
        if (markerIndex !== -1) {
          // Everything before the marker is sources (JSON)
          const sourcesPart = buffer.slice(0, markerIndex).trim();
          buffer = buffer.slice(markerIndex + LLM_SPLIT.length).trimStart();
          started = true;

          if (!sourcesEmitted) {
            try {
              const parsedSources = JSON.parse(sourcesPart);
              onSources(parsedSources);
            } catch (e) {
              console.error("Failed to parse sources:", e);
              onSources([]);
            }
            sourcesEmitted = true;
          }
        }
        // If no marker yet, discard buffer (still in sources part)
        else {
          buffer = "";
        }
      }

      // We are now in the LLM response part
      if (started) {
        const relatedIndex = buffer.indexOf(RELATED_SPLIT);

        if (relatedIndex !== -1) {
          // Take markdown up to related questions
          markdownBuffer += buffer.slice(0, relatedIndex);
          relatesBuffer = buffer.slice(relatedIndex + RELATED_SPLIT.length);

          // Send final markdown
          onMarkdown(markdownBuffer.trim());

          // Try to parse related questions
          try {
            const parsedRelates = JSON.parse(relatesBuffer);
            onRelates(parsedRelates);
          } catch (e) {
            console.error("Failed to parse related questions:", e);
            onRelates([]);
          }

          // Clear buffers
          buffer = "";
          markdownBuffer = "";
          relatesBuffer = "";
        } else {
          // Still in markdown part
          markdownBuffer += buffer;
          onMarkdown(markdownBuffer.trim());
          buffer = "";
        }
      }
    },
    () => {
      // End of stream - flush any remaining markdown
      if (markdownBuffer) {
        onMarkdown(markdownBuffer.trim());
      }
      // Final related questions parse if any leftover
      if (relatesBuffer) {
        try {
          const parsed = JSON.parse(relatesBuffer);
          onRelates(parsed);
        } catch {}
      }
    },
  );
};