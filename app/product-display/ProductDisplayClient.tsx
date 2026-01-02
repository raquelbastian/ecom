"use client";

import { useState, useEffect } from 'react';
import { useSearchParams } from 'next/navigation';
import Image from 'next/image';
import Link from 'next/link';
import {
  getRecommendations,
  getRecommendationsPCA,
  getRecommendationsReviews,
  getRecommendationsContent,
  getRecommendationsContentPCA,
  getRecommendationsSentiment,
  getRecommendationsTopic,
  getRecommendationsReviewerOverlap,
  getRecommendationsHybrid,
  getRecommendationsSVD,
  getRecommendationsKNN,
  getRecommendationsWeightedHybrid
} from '@/lib/mlService';
import Slider from 'react-slick';
import "slick-carousel/slick/slick.css"; 
import "slick-carousel/slick/slick-theme.css";

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

    const fetchAllRecommendations = async () => {
      setRecLoading(true);
      setRecError(null);
      setWeightedHybridRecs([]);
      try {
        // Fetch all recommendation types in parallel
        type RecommendationResponse = { recommendations: any[] };

        const [
          classic,
          pca,
          review,
          content,
          contentPca,
          sentiment,
          topic,
          reviewerOverlap,
          svd,
          knn,
          weighted
        ]: RecommendationResponse[] = await Promise.all([
          getRecommendations(selectedProduct.product_id, 5),
          getRecommendationsPCA(selectedProduct.product_id, 5),
          getRecommendationsReviews(selectedProduct.product_id, 5),
          getRecommendationsContent(selectedProduct.product_id, 5),
          getRecommendationsContentPCA(selectedProduct.product_id, 5),
          getRecommendationsSentiment(selectedProduct.product_id, 5),
          getRecommendationsTopic(selectedProduct.product_id, 5),
          getRecommendationsReviewerOverlap(selectedProduct.product_id, 5),
          getRecommendationsSVD(selectedProduct.product_id, 5),
          getRecommendationsKNN(selectedProduct.product_id, 5),
          getRecommendationsWeightedHybrid(selectedProduct.product_id, 5, {})
        ]);

        const allRecs = {
          classic_recs: classic?.recommendations || [],
          pca_recs: pca?.recommendations || [],
          review_recs: review?.recommendations || [],
          content_recs: content?.recommendations || [],
          content_pca_recs: contentPca?.recommendations || [],
          sentiment_recs: sentiment?.recommendations || [],
          topic_recs: topic?.recommendations || [],
          reviewer_overlap_recs: reviewerOverlap?.recommendations || [],
          svd_recs: svd?.recommendations || [],
          knn_recs: knn?.recommendations || [],
          recommendations: weighted?.recommendations || []
        };

        // Store all recommendations in session storage
        sessionStorage.setItem(`recs-${selectedProduct.product_id}`, JSON.stringify(allRecs));

        // Only set the weighted hybrid recs for display on this page
        setWeightedHybridRecs(allRecs.recommendations);

      } catch (err) {
        setRecError(String(err));
        setWeightedHybridRecs([]);
      } finally {
        setRecLoading(false);
      }
    };

    fetchAllRecommendations();
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

  const sliderSettings = {
    dots: true,
    infinite: false,
    speed: 500,
    slidesToShow: 4,
    slidesToScroll: 4,
    responsive: [
      {
        breakpoint: 1024,
        settings: {
          slidesToShow: 3,
          slidesToScroll: 3,
        }
      },
      {
        breakpoint: 600,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 2,
        }
      },
      {
        breakpoint: 480,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
        }
      }
    ]
  };

  return (
    <div>
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
        <div className="w-full max-w-2xl mt-8 flex justify-center">
        <div className="flex items-center gap-3">
        <svg className="animate-spin h-6 w-6 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
      </svg>
      <span className="text-gray-600">You may also like...</span>
    </div>
  </div>
) : recError ? (
          <div className="w-full max-w-2xl mt-8 text-red-600">Error loading recommendations: {recError}</div>
        ) : (
          <div className="w-full max-w-5xl mt-8 flex flex-col gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">You may also like</h3>
              <Slider {...sliderSettings}>
                {uniqueByProductId(weightedHybridRecs).map((r: any) => (
                  <div key={r.product_id} className="px-2">
                    <div className="border rounded-lg p-4 flex flex-col items-center text-center h-full">
                      <Link href={`/product-display?product_id=${encodeURIComponent(r.product_id)}`}>
                        <div className="relative w-40 h-40">
                          {r.img_link ? (
                            <Image src={r.img_link} alt={r.product_name} layout="fill" objectFit="contain" className="rounded-t-lg" unoptimized />
                          ) : <div className="w-full h-full bg-gray-200 rounded-t-lg"/>}
                        </div>
                      </Link>
                      <div className="mt-2 flex-grow flex flex-col justify-between">
                        <h4 className="font-semibold text-sm h-12 overflow-hidden">
                          <Link href={`/product-display?product_id=${encodeURIComponent(r.product_id)}`}>{r.product_name}</Link>
                        </h4>
                        <div>
                          <p className="mt-1 font-bold">${r.discounted_price}</p>
                          <p className="mt-1 text-xs text-gray-500">Rating: {r.rating ?? 'N/A'}</p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </Slider>
            </div>
          </div>
        )}
      </div>
      {selectedProduct && !recLoading && !recError && (
    <div className="mt-8 text-center">
    <Link href={`/product-display-recommend?product_id=${selectedProduct.product_id}`} className="text-blue-500 hover:underline">
      Check recommendation computation
    </Link>
  </div>
  )}
    </div>
  );
}