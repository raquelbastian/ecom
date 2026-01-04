"use client";

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Image from 'next/image';
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

export default function ProductDisplayClient() {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [classicRecs, setClassicRecs] = useState<any[]>([]);
  const [pcaRecs, setPcaRecs] = useState<any[]>([]);
  const [reviewRecs, setReviewRecs] = useState<any[]>([]);
  const [contentRecs, setContentRecs] = useState<any[]>([]);
  const [contentPcaRecs, setContentPcaRecs] = useState<any[]>([]);
  const [sentimentRecs, setSentimentRecs] = useState<any[]>([]);
  const [topicRecs, setTopicRecs] = useState<any[]>([]);
  const [reviewerOverlapRecs, setReviewerOverlapRecs] = useState<any[]>([]);
  const [hybridRecs, setHybridRecs] = useState<any[]>([]);
  const [svdRecs, setSvdRecs] = useState<any[]>([]);
  const [weightedHybridRecs, setWeightedHybridRecs] = useState<any[]>([]);
  const [knnRecs, setKnnRecs] = useState<Product[]>([]);
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
          setSelectedProduct(data as Product);
          setProducts([data as Product]);
        })
        .catch(err => {
          setProductError(String(err));
          setSelectedProduct(null);
        })
        .finally(() => setProductLoading(false));
    }
  }, [productId]);

  // Load all recommendations
  useEffect(() => {
    if (!selectedProduct) return;

    const pid = selectedProduct.product_id;
    setRecLoading(true);
    setRecError(null);

    const saveAndSet = (all: any) => {
      setClassicRecs(all.classic_recs || []);
      setPcaRecs(all.pca_recs || []);
      setReviewRecs(all.review_recs || []);
      setContentRecs(all.content_recs || []);
      setContentPcaRecs(all.content_pca_recs || []);
      setSentimentRecs(all.sentiment_recs || []);
      setTopicRecs(all.topic_recs || []);
      setReviewerOverlapRecs(all.reviewer_overlap_recs || []);
      setHybridRecs(all.hybrid_recs || []);
      setSvdRecs(all.svd_recs || []);
      setKnnRecs(all.knn_recs || []);
      setWeightedHybridRecs(all.recommendations || all.weighted || []);
    };

    (async () => {
      // 2) Try server-side cached JSON
      try {
        const res = await fetch(`/api/recs_cache?product_id=${encodeURIComponent(pid)}`);
        if (res.ok) {
          const payload = await res.json();
          if (payload && payload.recommendations) {
            saveAndSet(payload);
            setRecLoading(false);
            return;
          }
        }
      } catch (e) {
        // continue to fallback
      }

      // 3) Fallback: fetch weighted-hybrid quickly, render, then parallelize remaining models
      try {
        const ml = await import('@/lib/mlService');
        const weighted = await (ml as any).getRecommendationsWeightedHybrid(pid, 5, {});
        const initialAll: any = {
          classic_recs: [], pca_recs: [], review_recs: [], content_recs: [], content_pca_recs: [],
          sentiment_recs: [], topic_recs: [], reviewer_overlap_recs: [], svd_recs: [], knn_recs: [],
          recommendations: weighted?.recommendations || weighted || []
        };
        saveAndSet(initialAll);

        const results = await Promise.allSettled([
          (ml as any).getRecommendations(pid, 5),
          (ml as any).getRecommendationsPCA(pid, 5),
          (ml as any).getRecommendationsReviews(pid, 5),
          (ml as any).getRecommendationsContent(pid, 5),
          (ml as any).getRecommendationsContentPCA(pid, 5),
          (ml as any).getRecommendationsSentiment(pid, 5),
          (ml as any).getRecommendationsTopic(pid, 5),
          (ml as any).getRecommendationsReviewerOverlap(pid, 5),
          (ml as any).getRecommendationsSVD(pid, 5),
          (ml as any).getRecommendationsKNN(pid, 5)
        ]);

        const allRecs = {
          classic_recs: results[0].status === 'fulfilled' ? (results[0] as any).value?.recommendations || (results[0] as any).value : [],
          pca_recs: results[1].status === 'fulfilled' ? (results[1] as any).value?.recommendations || (results[1] as any).value : [],
          review_recs: results[2].status === 'fulfilled' ? (results[2] as any).value?.recommendations || (results[2] as any).value : [],
          content_recs: results[3].status === 'fulfilled' ? (results[3] as any).value?.recommendations || (results[3] as any).value : [],
          content_pca_recs: results[4].status === 'fulfilled' ? (results[4] as any).value?.recommendations || (results[4] as any).value : [],
          sentiment_recs: results[5].status === 'fulfilled' ? (results[5] as any).value?.recommendations || (results[5] as any).value : [],
          topic_recs: results[6].status === 'fulfilled' ? (results[6] as any).value?.recommendations || (results[6] as any).value : [],
          reviewer_overlap_recs: results[7].status === 'fulfilled' ? (results[7] as any).value?.recommendations || (results[7] as any).value : [],
          svd_recs: results[8].status === 'fulfilled' ? (results[8] as any).value?.recommendations || (results[8] as any).value : [],
          knn_recs: results[9].status === 'fulfilled' ? (results[9] as any).value?.recommendations || (results[9] as any).value : [],
          recommendations: initialAll.recommendations
        };

        // If weighted hybrid is empty, try to fill from any returned source
        if ((!allRecs.recommendations || allRecs.recommendations.length === 0) && initialAll.recommendations) {
          allRecs.recommendations = initialAll.recommendations;
        }

        saveAndSet(allRecs);
        setRecLoading(false);
        return;
      } catch (err) {
        console.error('Recommendation fallback failed', err);
        setRecError('Failed to load recommendations.');
        setRecLoading(false);
        return;
      }
    })();
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

  const renderTable = (title: string, data: any[]) => (
    <div>
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      {data.length > 0 ? (
        <table className="min-w-full border text-sm table-fixed">
          <thead>
            <tr>
              <th className="border px-2 py-1 w-20">Image</th>
              <th className="border px-2 py-1">Name</th>
              <th className="border px-2 py-1">Category</th>
              <th className="border px-2 py-1 w-24">Price</th>
              <th className="border px-2 py-1 w-20">Rating</th>
            </tr>
          </thead>
          <tbody>
            {uniqueByProductId(data).map((r: any) => (
              <tr key={r.product_id}>
                <td className="border px-2 py-1 align-top">
                  {r.img_link && <Image src={r.img_link} alt="" width={60} height={45} style={{ objectFit: 'cover', borderRadius: 4 }} unoptimized />}
                </td>
                <td className="border px-2 py-1 align-top max-w-[180px] truncate" title={r.product_name}><Link href={`/product-display?product_id=${encodeURIComponent(r.product_id)}`}>{r.product_name}</Link></td>
                <td className="border px-2 py-1 align-top max-w-[160px] truncate" title={r.category}>{r.category}</td>
                <td className="border px-2 py-1 align-top">${r.discounted_price}</td>
                <td className="border px-2 py-1 align-top">{r.rating ?? 'N/A'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p>No recommendations to display.</p>}
    </div>
  );

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">
      <h1 className="text-3xl font-bold mb-8 text-black dark:text-zinc-50">Product Details & Recommendation Computations</h1>
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
        <div className="w-full max-w-5xl mt-8 text-center">Loading recommendations...</div>
      ) : recError ? (
        <div className="w-full max-w-5xl mt-8 text-red-600 text-center">{recError}</div>
      ) : (
        <div className="w-full max-w-5xl mt-8 flex flex-col gap-8">
          {renderTable("Weighted Hybrid Recommendations", weightedHybridRecs)}
          {renderTable("Classic Recommendations", classicRecs)}
          {renderTable("PCA Recommendations", pcaRecs)}
          {renderTable("Review-Based Recommendations", reviewRecs)}
          {renderTable("Content-Based Recommendations", contentRecs)}
          {renderTable("Content-Based Recommendations with PCA", contentPcaRecs)}
          {renderTable("Sentiment-Based Recommendations", sentimentRecs)}
          {renderTable("Topic-Based Recommendations", topicRecs)}
          {renderTable("Reviewer Overlap Recommendations", reviewerOverlapRecs)}
          {renderTable("SVD Recommendations", svdRecs)}
          {renderTable("KNN Recommendations", knnRecs)}
        </div>
      )}
    </div>
  );
}