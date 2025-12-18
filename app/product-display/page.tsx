"use client";

import { useEffect, useState } from 'react';

interface Product {
  _id: string;
  product_id: string;
  product_name: string;
  category: string; 
  discounted_price: number;
  actual_price: number;
  discount_percentage: number;
  rating: number;
  rating_count: number;
  about_product: string;
  user_id: string;
  user_name: string;
  review_id: string;
  review_title: string;
  review_content: string;
  img_link: string;
  product_link: string;

}

export default function ProductDisplay() {
  const [products, setProducts] = useState<Product[]>([]);

  useEffect(() => {
    fetch('/api/products')
      .then(res => res.json())
      .then(data => {
        console.log('Fetched /api/products response:', data);
        if (Array.isArray(data)) {
          setProducts(data.slice(0, 10));
        } else if (data && data.products && Array.isArray(data.products)) {
          // handle responses that wrap products in an object
          setProducts(data.products.slice(0, 10));
        } else {
          console.error('Unexpected /api/products response format:', data);
          setProducts([]);
        }
      })
      .catch(err => {
        console.error('Error fetching products:', err);
        setProducts([]);
      });
  }, []);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">
      <h1 className="text-3xl font-bold mb-8 text-black dark:text-zinc-50">Product Listing</h1>
      <ul className="w-full max-w-2xl space-y-6">
        {products.map(product => (
          <li key={product._id} className="border p-6 rounded-lg bg-white dark:bg-zinc-900 shadow">
            <h2 className="text-xl font-semibold text-black dark:text-zinc-50">{product.product_name}</h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-2">{product.about_product}</p>
            <span className="text-lg font-bold text-blue-600 dark:text-blue-400">${product.actual_price}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
