// scripts/recreateVectorIndex.ts
// Drops and recreates the Atlas vector search index to match the Voyage AI
// embedding dimensions (voyage-3.5-lite → 1024). Run AFTER generateEmbeddings.ts.
import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

import clientPromise from '../lib/mongodb';

const INDEX_NAME = 'vector_index';
const NUM_DIMENSIONS = 1024;

async function recreateVectorIndex() {
  const client = await clientPromise;
  const collection = client.db().collection('products');

  const existing = await collection.listSearchIndexes(INDEX_NAME).toArray();
  if (existing.length > 0) {
    console.log(`Dropping existing "${INDEX_NAME}"...`);
    await collection.dropSearchIndex(INDEX_NAME);
    // Atlas takes a moment to delete; poll until gone
    while ((await collection.listSearchIndexes(INDEX_NAME).toArray()).length > 0) {
      await new Promise((r) => setTimeout(r, 3000));
      console.log('Waiting for old index to be deleted...');
    }
  }

  console.log(`Creating "${INDEX_NAME}" with ${NUM_DIMENSIONS} dimensions...`);
  await collection.createSearchIndex({
    name: INDEX_NAME,
    type: 'vectorSearch',
    definition: {
      fields: [
        {
          type: 'vector',
          path: 'embedding',
          numDimensions: NUM_DIMENSIONS,
          similarity: 'cosine',
        },
      ],
    },
  });

  // Wait until the index is queryable
  let status = '';
  while (status !== 'READY') {
    const [idx] = await collection.listSearchIndexes(INDEX_NAME).toArray();
    status = (idx as { status?: string })?.status ?? 'PENDING';
    console.log(`Index status: ${status}`);
    if (status !== 'READY') await new Promise((r) => setTimeout(r, 5000));
  }

  console.log('Vector index is READY — search is live on the new embeddings.');
  process.exit(0);
}

recreateVectorIndex();
