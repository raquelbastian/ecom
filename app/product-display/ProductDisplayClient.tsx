"use client";

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import { getRecommendationsWeightedHybrid } from '@/lib/mlService';

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

export default function ProductDisplayClient() {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [weightedHybridRecs, setWeightedHybridRecs] = useState<any[]>([]);
  const [productLoading, setProductLoading] = useState(false);
  const [productError, setProductError] = useState<string | null>(null);
  const [recLoading, setRecLoading] = useState(false);
  const [recError, setRecError] = useState<string | null>(null);
  const searchParams = useSearchParams();
  const productId = searchParams.get('product_id');

  useEffect(() => {
    if (productId) {
      setProductLoading(true);
      setProductError(null);
      fetch(`/api/products?product_id=${encodeURIComponent(productId)}`)
        .then(res => {
          if (!res.ok) throw new Error(`Product request failed: ${res.status}`);
          return res.json();
        })
        .then((data) => {
          console.log('Fetched single product response:', data);
          setSelectedProduct(data as Product);
          setProducts([data as Product]);
        })
        .catch(err => {
          console.error('Error fetching single product:', err);
          setProductError(String(err));
          setSelectedProduct(null);
        })
        .finally(() => setProductLoading(false));
      return;
    }

    // Fallback: fetch a short product list for navigation and selection
    setProductLoading(true);
    fetch('/api/products')
      .then(res => res.json())
      .then(data => {
        console.log('Fetched /api/products response:', data);
        const list = Array.isArray(data) ? data.slice(0, 10) : (data && data.products) || [];
        setProducts(list);
        if (productId) {
          const found = list.find((p: any) => p.product_id === productId);
          if (found) setSelectedProduct(found);
        }
      })
      .catch(err => {
        console.error('Error fetching products:', err);
        setProducts([]);
        setProductError(String(err));
      })
      .finally(() => setProductLoading(false));
  }, [productId]);

  // Automatically load recommendations when a product is selected
  useEffect(() => {
    if (!selectedProduct) return;

    const fetchWeightedHybridRecommendations = async () => {
      setRecLoading(true);
      setRecError(null);
      setWeightedHybridRecs([]);
      try {
        const res = await getRecommendationsWeightedHybrid(selectedProduct.product_id, 5, {}); // pass empty object for default
        setWeightedHybridRecs(res && res.recommendations ? res.recommendations : []);
      } catch (err) {
        setRecError(String(err));
        setWeightedHybridRecs([]);
      } finally {
        setRecLoading(false);
      }
    };

    fetchWeightedHybridRecommendations();
  }, [selectedProduct]);

  // Utility to filter unique product_ids in an array
  function uniqueByProductId(arr: any[]) {
    const seen = new Set();
    return arr.filter((item) => {
      if (!item.product_id || seen.has(item.product_id)) return false;
      seen.add(item.product_id);
      return true;
    });
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">
      <h1 className="text-3xl font-bold mb-8 text-black dark:text-zinc-50">Product Details</h1>
      {productLoading ? (
        <div className="w-full max-w-2xl border p-6 rounded-lg bg-white dark:bg-zinc-900 shadow text-center">Loading product...</div>
      ) : productError ? (
        <div className="w-full max-w-2xl border p-6 rounded-lg bg-white dark:bg-zinc-900 shadow text-red-600">Error loading product: {productError}</div>
      ) : selectedProduct ? (
        <div className="w-full max-w-2xl border p-6 rounded-lg bg-white dark:bg-zinc-900 shadow">
          <h2 className="text-2xl font-bold">{selectedProduct.product_name}</h2>
          <div className="flex items-center mb-2">
            <span className="text-yellow-500 font-semibold mr-2">Rating:</span>
            <span className="text-black dark:text-zinc-100">{selectedProduct.rating ?? 'N/A'}</span>
          </div>
          <p className="text-zinc-600 dark:text-zinc-400 mb-4">{selectedProduct.about_product}</p>
          {selectedProduct.img_link ? (
            <div className="mb-4 w-full relative" style={{ maxWidth: 400, height: 300 }}>
              <Image src={selectedProduct.img_link} alt={selectedProduct.product_name} fill style={{ objectFit: 'cover' }} unoptimized />
              <img src={selectedProduct.img_link} alt={selectedProduct.product_name} style={{ maxWidth: '100%', display: 'none' }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }} />
            </div>
          ) : null}
          <div>
            <strong className="text-lg mr-4">Price: ${selectedProduct.actual_price}</strong>
          </div>
        </div>
      ) : (
        <p>No product selected.</p>
      )}

      {recLoading ? (
        <div className="w-full max-w-2xl mt-8">Loading recommendations...</div>
      ) : recError ? (
        <div className="w-full max-w-2xl mt-8 text-red-600">Error loading recommendations: {recError}</div>
      ) : (
        <div className="w-full max-w-5xl mt-8 flex flex-col gap-8">
          {/* Weighted Hybrid Recommendations */}
          <div>
            <h3 className="text-xl font-semibold mb-4">Weighted Hybrid Recommendations</h3>
            <table className="min-w-full border text-sm table-fixed">
              <colgroup>
                <col style={{ width: '80px' }} />
                <col style={{ width: '180px' }} />
                <col style={{ width: '160px' }} />
                <col style={{ width: '80px' }} />
                <col style={{ width: '60px' }} />
              </colgroup>
              <thead>
                <tr className="bg-zinc-100 dark:bg-zinc-800">
                  <th className="border px-2 py-1">Image</th>
                  <th className="border px-2 py-1">Product Name</th>
                  <th className="border px-2 py-1">Category</th>
                  <th className="border px-2 py-1">Price</th>
                  <th className="border px-2 py-1">Rating</th>
                </tr>
              </thead>
              <tbody>
                {uniqueByProductId(weightedHybridRecs).map((r: any) => (
                  <tr key={r.product_id}>
                    <td className="border px-2 py-1 align-top">
                      {r.img_link ? (
                        <Image src={r.img_link} alt="" width={60} height={45} style={{ objectFit: 'cover', borderRadius: 4 }} unoptimized />
                      ) : null}
                    </td>
                    <td className="border px-2 py-1 align-top max-w-[180px] truncate" title={r.product_name} style={{ wordBreak: 'break-word', whiteSpace: 'pre-line' }}>
                      <Link href={`/product-display?product_id=${encodeURIComponent(r.product_id)}`}>{r.product_name}</Link>
                    </td>
                    <td className="border px-2 py-1 align-top max-w-[160px] truncate" title={r.category} style={{ wordBreak: 'break-word', whiteSpace: 'pre-line' }}>
                      {r.category}
                    </td>
                    <td className="border px-2 py-1 align-top">${r.discounted_price}</td>
                    <td className="border px-2 py-1 align-top">{r.rating ?? 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
