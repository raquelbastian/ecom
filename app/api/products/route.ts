import { NextResponse } from 'next/server';
import clientPromise from '../../../lib/mongodb';

export async function GET() {
  try {
    const client = await clientPromise;
    const db = client.db();
    const products = await db.collection('products').find({}).toArray() as any[];

    // Convert ObjectId to string and normalize field names for the frontend
    const serializable = products.map((p: any) => ({
      ...p,
      _id: p._id?.toString?.(),
      discounted_price: p.discounted_price ?? p.discountedPrice ?? p.price ?? null,
      actual_price: p.actual_price ?? p.actualPrice ?? null,
      product_name: p.product_name ?? p.name ?? null,
      about_product: p.about_product ?? p.description ?? '',
    }));

    return NextResponse.json(serializable);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch products' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';