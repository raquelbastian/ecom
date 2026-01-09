import { NextResponse } from 'next/server';
import { getProducts } from '../../../lib/mongodb';

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const category = url.searchParams.get('category') || undefined;
    const productId = url.searchParams.get('product_id') || undefined;
    const search = url.searchParams.get('search') || undefined;
    const limitParam = url.searchParams.get('limit');
    const limit = limitParam ? Math.min(100, Number(limitParam)) : undefined;
    const rand = url.searchParams.get('rand');
    const randomize = rand !== null;
    const vectorSearch = url.searchParams.get('vectorSearch') === 'true';

    const products = await getProducts({ category, productId, search, limit, randomize, vectorSearch });

    if (productId && !products) {
      return NextResponse.json(null, { status: 404 });
    }

    return NextResponse.json(products);
  } catch (error) {
    console.error('Error fetching products:', error);
    return NextResponse.json({ error: 'Failed to fetch products' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
export const maxDuration = 60; // Increase timeout to 60 seconds