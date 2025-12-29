// MongoDB connection utility for Next.js
import { MongoClient } from 'mongodb';

declare global {
  // eslint-disable-next-line no-var
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

const uri = process.env.MONGODB_URI || '';
const options = {};

let client: MongoClient;
let clientPromise: Promise<MongoClient>;

if (!process.env.MONGODB_URI) {
  throw new Error('Please add your MongoDB URI to .env.local');
}

if (process.env.NODE_ENV === 'development') {
  // In development mode, use a global variable so that the value
  // is preserved across module reloads caused by HMR (Hot Module Replacement).
  if (!global._mongoClientPromise) {
    client = new MongoClient(uri, options);
    global._mongoClientPromise = client.connect();
  }
  clientPromise = global._mongoClientPromise as Promise<MongoClient>;
} else {
  // In production mode, it's best to not use a global variable.
  client = new MongoClient(uri, options);
  clientPromise = client.connect()
}

export default clientPromise;

export async function getProducts(options: { category?: string; limit?: number; randomize?: boolean; productId?: string }) {
  const { category, limit = 12, randomize = false, productId } = options;
  const client = await clientPromise;
  const db = client.db();
  const productsCollection = db.collection('products');

  // If a single product_id is requested, return that product directly
  if (productId) {
    const product = await productsCollection.findOne({ product_id: productId });
    if (!product) return null;
    const serialized = {
      ...product,
      _id: product._id?.toString?.(),
      discounted_price: product.discounted_price ?? product.discountedPrice ?? product.price ?? null,
      actual_price: product.actual_price ?? product.actualPrice ?? null,
      product_name: product.product_name ?? product.name ?? null,
      about_product: product.about_product ?? product.description ?? '',
    };
    return serialized;
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

  let products;

  if (randomize) {
    const pipeline = [
      { $match: query },
      { $sample: { size: limit } }
    ];
    products = await productsCollection.aggregate(pipeline).toArray();
  } else {
    let cursor = productsCollection.find(query);
    if (limit > 0) {
      cursor = cursor.limit(limit);
    }
    products = await cursor.toArray();
  }

  const serializable = (products as any[]).map((p: any) => ({
    ...p,
    _id: p._id?.toString?.(),
    discounted_price: p.discounted_price ?? p.discountedPrice ?? p.price ?? null,
    actual_price: p.actual_price ?? p.actualPrice ?? null,
    product_name: p.product_name ?? p.name ?? null,
    about_product: p.about_product ?? p.description ?? '',
  }));

  return serializable;
}
