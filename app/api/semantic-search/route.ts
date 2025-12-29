import { NextRequest, NextResponse } from 'next/server';
import { OpenAI } from 'openai';
import { getProducts } from '@/lib/mongodb';

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

export async function POST(req: NextRequest) {
  const { query } = await req.json();

  // Use GPT to extract filters from the query
  const prompt = `
You are an e-commerce search assistant. Given a user query, extract the main category, price range, and any important features or keywords. 
Return a JSON object with keys: category, min_price, max_price, keywords (array of strings).
User query: "${query}"
`;

  const gptRes = await openai.chat.completions.create({
    model: 'gpt-3.5-turbo',
    messages: [{ role: 'user', content: prompt }],
    temperature: 0.2,
  });

  let filters = {};
  try {
    filters = JSON.parse(gptRes.choices[0].message.content || '{}');
  } catch {
    filters = {};
  }

  // Fetch all products (or filter in DB if possible)
  const allProducts = await getProducts();

  // Filter products in-memory (for demo; for production, filter in DB)
  let results = allProducts;
  if (filters.category) {
    results = results.filter((p: any) =>
      p.category?.toLowerCase().includes(filters.category.toLowerCase())
    );
  }
  if (filters.min_price || filters.max_price) {
    results = results.filter((p: any) => {
      const price = parseFloat(p.discounted_price);
      if (filters.min_price && price < filters.min_price) return false;
      if (filters.max_price && price > filters.max_price) return false;
      return true;
    });
  }
  if (filters.keywords && filters.keywords.length > 0) {
    results = results.filter((p: any) =>
      filters.keywords.some((kw: string) =>
        (p.product_name + ' ' + (p.about_product || '')).toLowerCase().includes(kw.toLowerCase())
      )
    );
  }

  // Limit to top 20 results
  results = results.slice(0, 20);

  return NextResponse.json({ results });
}
