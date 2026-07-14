import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { getProducts } from '../../../lib/mongodb';

// RAG shopping assistant (multi-turn):
//   R — retrieve products via the existing vector search (Voyage + Atlas)
//       for the LATEST user message
//   A — augment that message with the retrieved product details; earlier
//       turns stay in the conversation, so follow-ups can refer to
//       products retrieved in previous turns
//   G — Claude generates an answer grounded ONLY in retrieved products

const SYSTEM_PROMPT = `You are the shopping assistant for an online store. You help customers find and choose products from the store's catalog through a chat conversation.

Rules:
- Base your answers ONLY on retrieved products shown in this conversation (the current message and earlier ones). Never invent, assume, or mention products that were not retrieved.
- Evaluate all retrieved products equally, regardless of their order in the list.
- Base your assessment on each product's actual rating and details, even if it disagrees with how the customer framed their question.
- Quote prices exactly as they appear in the product data.
- For follow-up questions (e.g. "is there a cheaper one?", "compare those two"), use the products already retrieved in this conversation when they answer the question.
- If none of the retrieved products genuinely fit, say so honestly — do not force a recommendation.
- Reply in the same language the customer used (English, Tagalog, or Taglish).
- Keep replies short and friendly: 2–4 sentences. Plain conversational text only — no markdown of any kind (no **bold**, no bullets, no tables, no headers); the chat UI renders plain text.`;

const MAX_HISTORY = 12; // cap context growth

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export async function POST(request: Request) {
  let messages: ChatMessage[];
  try {
    const body = await request.json();
    // Accept either { messages: [...] } (chat) or { query } (single-turn)
    if (Array.isArray(body?.messages) && body.messages.length > 0) {
      messages = body.messages;
    } else if (typeof body?.query === 'string' && body.query.trim()) {
      messages = [{ role: 'user', content: body.query }];
    } else {
      return NextResponse.json({ error: 'messages or query is required' }, { status: 400 });
    }
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  const isValid = messages.every(
    (m) => (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string'
  );
  const last = messages[messages.length - 1];
  if (!isValid || last.role !== 'user') {
    return NextResponse.json({ error: 'Last message must be from the user' }, { status: 400 });
  }

  messages = messages.slice(-MAX_HISTORY);

  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn('ANTHROPIC_API_KEY is not set; skipping AI answer.');
    return NextResponse.json({ answer: null }, { status: 200 });
  }

  // R — Retrieval for the latest user message
  let products: any[] = [];
  try {
    const result = await getProducts({
      search: messages[messages.length - 1].content,
      vectorSearch: true,
      limit: 6,
    });
    products = Array.isArray(result) ? result : [];
  } catch (err) {
    console.error('Assistant retrieval failed:', err);
  }

  // A — Augmentation: attach fresh retrieval to the latest user message.
  // Earlier turns already carry their own retrieved products, so the
  // conversation stays grounded across follow-ups.
  const context =
    products.length > 0
      ? products
          .map(
            (p, i) =>
              `${i + 1}. ${p.product_name}\n` +
              `   Price: ${p.discounted_price} (original: ${p.actual_price})\n` +
              `   Rating: ${p.rating} stars (${p.rating_count} ratings)\n` +
              `   About: ${String(p.about_product ?? '').slice(0, 300)}`
          )
          .join('\n\n')
      : '(no additional products retrieved for this message)';

  const claudeMessages = messages.map((m, idx) =>
    idx === messages.length - 1
      ? {
          role: 'user' as const,
          content:
            `Retrieved products from the catalog for this message:\n\n${context}\n\n` +
            `Customer's message: "${m.content}"`,
        }
      : { role: m.role, content: m.content }
  );

  // G — Generation, grounded in retrieved products
  try {
    const anthropic = new Anthropic();
    const response = await anthropic.messages.create({
      model: 'claude-opus-4-8',
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: claudeMessages,
    });

    const answer = response.content
      .filter((block) => block.type === 'text')
      .map((block) => (block as { type: 'text'; text: string }).text)
      .join('');

    return NextResponse.json({ answer: answer || null });
  } catch (err) {
    console.error('Assistant generation failed:', err);
    return NextResponse.json({ answer: null }, { status: 200 });
  }
}

export const dynamic = 'force-dynamic';
