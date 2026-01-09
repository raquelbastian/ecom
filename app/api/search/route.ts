import { NextResponse } from 'next/server';
import clientPromise from '../../../lib/mongodb';
import { getEmbedding } from '../../../lib/embedding';

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const query = url.searchParams.get('query');

    if (!query) {
      return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
    }

    const client = await clientPromise;
    const db = client.db();
    const productsCollection = db.collection('products');

    console.log(`Performing vector search for: "${query}"`);
    const searchEmbedding = await getEmbedding(query);
    console.log('Generated search embedding.');

    const pipeline = [
      {
        $vectorSearch: {
          index: 'vector_index',
          queryVector: searchEmbedding,
          path: 'embedding',
          numCandidates: 100,
          limit: 12,
        },
      },
      {
        $project: {
          _id: 0,
          product_id: 1,
          name: 1,
          description: 1,
          image: 1,
          price: 1,
          score: { $meta: 'vectorSearchScore' },
        },
      },
    ];

    console.log('Executing vector search pipeline...');
    const products = await productsCollection.aggregate(pipeline).toArray();
    console.log(`Found ${products.length} products from vector search.`);

    return NextResponse.json(products);
  } catch (error) {
    console.error('Error performing vector search:', error);
    return NextResponse.json({ error: 'Failed to perform vector search' }, { status: 500 });
  }
}

export const dynamic = 'force-dynamic';
