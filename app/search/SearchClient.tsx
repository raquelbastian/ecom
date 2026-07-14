"use client";

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import ProductGrid from '@/components/ProductGrid';
import Link from 'next/link';

interface Product {
  product_id: string;
  product_name: string;
  category: string;
  actual_price: string;
  discounted_price: string;
  rating: string;
  rating_count: string;
  about_product: string;
  img_link: string;
}

export default function SearchClient() {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [aiAnswer, setAiAnswer] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);
  const searchParams = useSearchParams();
  const query = searchParams.get('query');

  useEffect(() => {
    if (!query) {
      setProducts([]);
      setLoading(false);
      return;
    }

    const fetchProducts = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(`/api/products?search=${encodeURIComponent(query)}&vectorSearch=true`);
        if (!res.ok) {
          throw new Error('Failed to fetch search results');
        }
        const data = await res.json();
        setProducts(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setLoading(false);
      }
    };

    // RAG answer — runs in parallel with the grid fetch; failures are silent
    // so the search experience never depends on the AI layer
    const fetchAiAnswer = async () => {
      setAiAnswer(null);
      setAiLoading(true);
      try {
        const res = await fetch('/api/assistant', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        });
        if (res.ok) {
          const data = await res.json();
          setAiAnswer(data.answer ?? null);
        }
      } catch {
        // silent — grid still works without the AI answer
      } finally {
        setAiLoading(false);
      }
    };

    fetchProducts();
    fetchAiAnswer();
  }, [query]);

  return (
    <div className="container mx-auto p-4">
      <Link href="/" className="text-blue-500 hover:underline mb-4 block">&larr; Back to Home</Link>
      <h1 className="text-2xl font-bold mb-4">Search Results for "{query}"</h1>

      {(aiLoading || aiAnswer) && (
        <div className="mb-6 rounded-lg border border-indigo-200 bg-indigo-50 p-4">
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-indigo-500">
            ✨ AI Shopping Assistant
          </p>
          {aiLoading ? (
            <p className="animate-pulse text-sm text-indigo-400">Looking at the results for you...</p>
          ) : (
            <p className="whitespace-pre-line text-sm text-gray-800">{aiAnswer}</p>
          )}
        </div>
      )}

      {loading && <p>Loading...</p>}
      {error && <p className="text-red-500">Error: {error}</p>}
      {!loading && !error && (
        <>
          {products.length > 0 ? (
            <ProductGrid products={products} />
          ) : (
            <p>No products found.</p>
          )}
        </>
      )}
    </div>
  );
}
