import { NextResponse } from 'next/server';
import { getProducts } from '../../../lib/mongodb';

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const category = url.searchParams.get('category') || undefined;
    const productId = url.searchParams.get('product_id') || undefined;
    const limitParam = url.searchParams.get('limit');
    const limit = limitParam ? Math.min(100, Number(limitParam)) : undefined;
    const rand = url.searchParams.get('rand');
    const randomize = rand !== null;

    const products = await getProducts({ category, productId, limit, randomize });

    if (productId && !products) {
      return NextResponse.json(null, { status: 404 });
    }

    return NextResponse.json(products);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch products' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';