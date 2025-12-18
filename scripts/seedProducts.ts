// Example product seeding script for MongoDB
import { MongoClient } from 'mongodb';
import * as dotenv from 'dotenv';
import path from 'path';

dotenv.config({ path: path.resolve(process.cwd(), '.env.local') });

const uri = process.env.MONGODB_URI || '';
console.log('DEBUG MONGODB_URI:', uri); // Debug print
const client = new MongoClient(uri);

async function seedProducts() {
  try {
    await client.connect();
    const db = client.db();
    const products = db.collection('products');
    await products.deleteMany({});
    await products.insertMany([
      { name: 'T-shirt', price: 19.99, description: 'Comfortable cotton T-shirt' },
      { name: 'Jeans', price: 49.99, description: 'Stylish blue jeans' },
      { name: 'Sneakers', price: 89.99, description: 'Running sneakers' },
    ]);
    console.log('Products seeded!');
  } finally {
    await client.close();
  }
}

seedProducts();
