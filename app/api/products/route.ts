import { NextResponse } from 'next/server';
import clientPromise from '../../../lib/mongodb';

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const category = url.searchParams.get('category');
    const productId = url.searchParams.get('product_id');
    const limitParam = url.searchParams.get('limit');
    const limit = limitParam ? Math.min(100, Number(limitParam)) : 0;

    const client = await clientPromise;
    const db = client.db();

    // If a single product_id is requested, return that product directly
    if (productId) {
      const product = await db.collection('products').findOne({ product_id: productId });
      if (!product) return NextResponse.json(null, { status: 404 });
      const serialized = {
        ...product,
        _id: product._id?.toString?.(),
        discounted_price: product.discounted_price ?? product.discountedPrice ?? product.price ?? null,
        actual_price: product.actual_price ?? product.actualPrice ?? null,
        product_name: product.product_name ?? product.name ?? null,
        about_product: product.about_product ?? product.description ?? '',
      };
      return NextResponse.json(serialized);
    }

    // Build query. If `category` is supplied, try to match either an array field `categories`
    // (preferred long-term) or match a token inside the pipe-delimited `category` string.
    const query: any = {};
    if (category) {
      // Normalize incoming category: remove whitespace so "Computers & Accessories" -> "Computers&Accessories"
      const normalized = category.replace(/\s+/g, '');

      // Escape regex meta-characters for safe regex construction
      const escapeRegExp = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const esc = escapeRegExp(normalized);

      // Regex to match the token as a whole entry in a pipe-delimited string (start, between pipes, or end)
      const tokenRegex = `(^|\\|)${esc}(\\||$)`;

      query.$or = [
        // If documents are migrated to a `categories` array, match by exact membership
        { categories: { $in: [normalized] } },
        // Otherwise match inside pipe-delimited `category` string (case-insensitive)
        { category: { $regex: tokenRegex, $options: 'i' } },
      ];
    }

    // If a randomization param is present, randomize the order
    const rand = url.searchParams.get('rand');
    let cursor = db.collection('products').find(query);
    if (rand) {
      // Use MongoDB's $sample aggregation for randomization
      const sampleSize = limit && limit > 0 ? limit : 12;
      const pipeline = [
        { $match: query },
        { $sample: { size: sampleSize } }
      ];
      const products = await db.collection('products').aggregate(pipeline).toArray();
      const serializable = (products as any[]).map((p: any) => ({
        ...p,
        _id: p._id?.toString?.(),
        discounted_price: p.discounted_price ?? p.discountedPrice ?? p.price ?? null,
        actual_price: p.actual_price ?? p.actualPrice ?? null,
        product_name: p.product_name ?? p.name ?? null,
        about_product: p.about_product ?? p.description ?? '',
      }));
      return NextResponse.json(serializable);
    }

    if (limit && limit > 0) cursor = cursor.limit(limit);

    const products = await cursor.toArray();

    const serializable = (products as any[]).map((p: any) => ({
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