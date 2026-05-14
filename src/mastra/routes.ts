import { registerApiRoute } from '@mastra/core/server';

// Default fallback when the agent doesn't yet carry routing decisions.
// Update these when actual provider routing lands.
const DEFAULT_MODEL = 'claude-sonnet-4-6';
const DEFAULT_REASON = 'general assistant';

type ChatMessage = {
  role: 'user' | 'assistant' | 'system';
  content: string;
};

// Shape of an optional file attachment passed from FastAPI. We preserve
// image + PDF uploads by translating to multimodal content parts that the
// underlying Claude model can read.
type AttachedFile = {
  type: string; // mime type
  base64: string; // raw base64 (no data: prefix)
};

type ContentPart =
  | { type: 'text'; text: string }
  | { type: 'image'; image: string; mimeType?: string }
  | { type: 'file'; data: string; mimeType?: string; mediaType?: string };

function buildUserContent(
  message: string,
  file: AttachedFile | undefined
): string | ContentPart[] {
  if (!file || !file.base64) return message;

  const parts: ContentPart[] = [];
  if (file.type === 'application/pdf') {
    parts.push({
      type: 'file',
      data: file.base64,
      mimeType: 'application/pdf',
      mediaType: 'application/pdf',
    });
  } else if (file.type.startsWith('image/')) {
    parts.push({
      type: 'image',
      image: file.base64,
      mimeType: file.type,
    });
  }
  parts.push({ type: 'text', text: message });
  return parts;
}

export const chatStreamRoute = registerApiRoute('/homeai/chat/stream', {
  method: 'POST',
  handler: async (c) => {
    const internalSecret = c.req.header('x-internal-auth');
    if (!internalSecret || internalSecret !== process.env.INTERNAL_SECRET) {
      return c.json({ error: 'Unauthorized' }, 401);
    }

    const body = await c.req.json();
    const { agentId, message, resourceId, threadId, conversationHistory, file } =
      body as {
        agentId?: string;
        message?: string;
        resourceId?: string;
        threadId?: string;
        conversationHistory?: ChatMessage[];
        file?: AttachedFile;
      };

    if (!agentId || !message || !resourceId || !threadId) {
      return c.json(
        {
          error: 'Missing required fields: agentId, message, resourceId, threadId',
        },
        400
      );
    }

    const mastra = c.get('mastra');
    const agent = mastra.getAgent(agentId);

    if (!agent) {
      return c.json({ error: `Agent not found: ${agentId}` }, 404);
    }

    // FastAPI passes short-term history; Mastra memory handles long-term recall
    // via the (resource, thread) pair.
    const priorMessages: ChatMessage[] = Array.isArray(conversationHistory)
      ? conversationHistory
          .filter(
            (m): m is ChatMessage =>
              !!m &&
              typeof m === 'object' &&
              (m.role === 'user' || m.role === 'assistant' || m.role === 'system') &&
              typeof m.content === 'string'
          )
          .map((m) => ({ role: m.role, content: m.content }))
      : [];

    const userContent = buildUserContent(message, file);

    // Use frontend-provided history as the sole source of conversation context.
    // Mastra thread memory (workingMemory + lastMessages) was injecting the full
    // prior conversation on top of priorMessages, causing every previous answer
    // to appear twice — the model then regenerated old answers before answering
    // the new question. Dropping the memory option gives us clean, controlled context.
    const messages: any[] = [
      ...priorMessages,
      { role: 'user', content: userContent },
    ];

    const stream = await agent.stream(messages);

    return new Response(
      new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder();
          const send = (payload: Record<string, unknown>) => {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
          };

          try {
            for await (const chunk of stream.textStream) {
              send({ type: 'text-delta', delta: chunk });
            }
            send({
              type: 'done',
              model: DEFAULT_MODEL,
              reason: DEFAULT_REASON,
            });
          } catch (err) {
            const errMsg = err instanceof Error ? err.message : 'Unknown error';
            send({ type: 'error', error: errMsg });
          } finally {
            controller.close();
          }
        },
      }),
      {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
      }
    );
  },
});
