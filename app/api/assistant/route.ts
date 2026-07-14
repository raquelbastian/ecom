import { NextResponse } from 'next/server';
import Anthropic from '@anthropic-ai/sdk';
import { getProducts } from '../../../lib/mongodb';

// RAG search assistant:
//   R — retrieve products via the existing vector search (Voyage + Atlas)
//   A — augment the prompt with the retrieved product details
//   G — Claude generates an answer grounded ONLY in those products

const SYSTEM_PROMPT = `You are the shopping assistant for an online store. You help customers find products from the store's catalog.

Rules:
- Base your answer ONLY on the retrieved products provided in the message. Never invent, assume, or mention products that are not in the list.
- Evaluate all retrieved products equally, regardless of their order in the list.
- Base your assessment on each product's actual rating and details, even if it disagrees with how the customer framed their question.
- Quote prices exactly as they appear in the product data.
- If none of the retrieved products genuinely fit what the customer is looking for, say so honestly — do not force a recommendation.
- Reply in the same language the customer used (English, Tagalog, or Taglish).
- Keep it short and friendly: 2–4 sentences highlighting the best matches and why. No markdown tables, no headers — plain conversational text.`;

export async function POST(request: Request) {
  let query: string;
  try {
    const body = await request.json();
    query = body?.query;
  } catch {
    return NextResponse.json({ error: 'Invalid JSON body' }, { status: 400 });
  }

  if (!query || typeof query !== 'string') {
    return NextResponse.json({ error: 'query is required' }, { status: 400 });
  }

  // R — Retrieval (reuses the same vector search that powers the results grid)
  let products: any[] = [];
  try {
    const result = await getProducts({ search: query, vectorSearch: true, limit: 6 });
    products = Array.isArray(result) ? result : [];
  } catch (err) {
    console.error('Assistant retrieval failed:', err);
    return NextResponse.json({ answer: null }, { status: 200 });
  }

  if (products.length === 0) {
    return NextResponse.json({ answer: null }, { status: 200 });
  }

  if (!process.env.ANTHROPIC_API_KEY) {
    // No key configured — search still works, we just skip the AI answer
    console.warn('ANTHROPIC_API_KEY is not set; skipping AI answer.');
    return NextResponse.json({ answer: null }, { status: 200 });
  }

  // A — Augmentation: hydrated product details go into the prompt
  const context = products
    .map(
      (p, i) =>
        `${i + 1}. ${p.product_name}\n` +
        `   Price: ${p.discounted_price} (original: ${p.actual_price})\n` +
        `   Rating: ${p.rating} stars (${p.rating_count} ratings)\n` +
        `   About: ${String(p.about_product ?? '').slice(0, 300)}`
    )
    .join('\n\n');

  // G — Generation, grounded in the retrieved products
  try {
    const anthropic = new Anthropic();
    const response = await anthropic.messages.create({
      model: 'claude-opus-4-8',
      max_tokens: 1024,
      system: SYSTEM_PROMPT,
      messages: [
        {
          role: 'user',
          content:
            `Retrieved products from the catalog for this search:\n\n${context}\n\n` +
            `Customer's search: "${query}"\n\n` +
            `Help the customer choose from these products.`,
        },
      ],
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
