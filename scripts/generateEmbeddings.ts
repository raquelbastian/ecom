// scripts/generateEmbeddings.ts
import dotenv from 'dotenv';
dotenv.config({ path: '.env.local' });

import { getProducts } from '../lib/mongodb';
import { getEmbedding } from '../lib/embedding';
import clientPromise from '../lib/mongodb';
import { ObjectId } from 'mongodb';

async function generateEmbeddings() {
  const products = await getProducts({ limit: 0 }); // Fetch all products
  const client = await clientPromise;
  const db = client.db();
  const productsCollection = db.collection('products');

  if (!Array.isArray(products)) {
    console.error('Could not retrieve products.');
    return;
  }

  for (const product of products) {
    const textToEmbed = `${product.product_name} ${product.about_product}`;
    const embedding = await getEmbedding(textToEmbed);

    await productsCollection.updateOne(
      { _id: new ObjectId(product._id) },
      { $set: { embedding: Array.from(embedding) } }
    );
    console.log(`Generated and stored embedding for ${product.product_name}`);
  }

  console.log('Finished generating embeddings for all products.');
  process.exit(0);
}

generateEmbeddings();
