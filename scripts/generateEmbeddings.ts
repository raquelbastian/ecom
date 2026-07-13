// scripts/generateEmbeddings.ts
import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

import { getProducts } from '../lib/mongodb';
import { getEmbeddings } from '../lib/embedding';
import clientPromise from '../lib/mongodb';
import { ObjectId } from 'mongodb';

// Batch size and pacing sized for Voyage's no-payment-method free tier
// (3 requests/min, 10K tokens/min). With a payment method on file the limits
// are much higher and the waits just make the run slower than necessary —
// feel free to raise BATCH_SIZE to 128 and drop INTER_BATCH_WAIT_MS to 0.
const BATCH_SIZE = 16;
const INTER_BATCH_WAIT_MS = 30_000;
const RATE_LIMIT_WAIT_MS = 30_000;
const MAX_RETRIES = 8;
const EXPECTED_DIMENSIONS = 1024;

async function embedBatchWithRetry(texts: string[], attempt = 1): Promise<number[][]> {
  try {
    return await getEmbeddings(texts, 'document');
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (message.includes('429') && attempt <= MAX_RETRIES) {
      console.log(`Rate limited — waiting ${RATE_LIMIT_WAIT_MS / 1000}s before retry ${attempt}/${MAX_RETRIES}...`);
      await new Promise((r) => setTimeout(r, RATE_LIMIT_WAIT_MS));
      return embedBatchWithRetry(texts, attempt + 1);
    }
    throw err;
  }
}

async function generateEmbeddings() {
  const allProducts = await getProducts({ limit: 0 }); // Fetch all products
  const client = await clientPromise;
  const db = client.db();
  const productsCollection = db.collection('products');

  if (!Array.isArray(allProducts)) {
    console.error('Could not retrieve products.');
    return;
  }

  // Resume support: skip products that already have an embedding with the
  // expected dimensions (from a previous partial run)
  const products = allProducts.filter(
    (p) => !(Array.isArray(p.embedding) && p.embedding.length === EXPECTED_DIMENSIONS)
  );
  console.log(
    `${allProducts.length} products total, ${allProducts.length - products.length} already embedded — embedding remaining ${products.length} in batches of ${BATCH_SIZE}...`
  );

  for (let i = 0; i < products.length; i += BATCH_SIZE) {
    const batch = products.slice(i, i + BATCH_SIZE);
    const texts = batch.map((p) => `${p.product_name} ${p.about_product}`);

    const embeddings = await embedBatchWithRetry(texts);

    await productsCollection.bulkWrite(
      batch.map((product, j) => ({
        updateOne: {
          filter: { _id: new ObjectId(product._id) },
          update: { $set: { embedding: embeddings[j] } },
        },
      }))
    );

    console.log(`Stored embeddings ${i + 1}–${i + batch.length} of ${products.length}`);

    if (i + BATCH_SIZE < products.length) {
      await new Promise((r) => setTimeout(r, INTER_BATCH_WAIT_MS));
    }
  }

  console.log('Finished generating embeddings for all products.');
  process.exit(0);
}

generateEmbeddings();
