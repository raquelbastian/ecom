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
import ProductGrid from '@/components/ProductGrid';

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
  const [recLatencyMs, setRecLatencyMs] = useState<number | null>(null);
  const searchParams = useSearchParams();
  const productId = searchParams.get('product_id');
  const search = searchParams.get('search');

  useEffect(() => {
    const fetchProducts = async () => {
      setProductLoading(true);
      setProductError(null);
      try {
        let url = '/api/products';
        const queryParams = new URLSearchParams();
        if (productId) {
          queryParams.append('product_id', productId);
        } else if (search) {
          queryParams.append('search', search);
          queryParams.append('vectorSearch', 'true');
        } else {
          // By default, fetch a few random products
          queryParams.append('limit', '12');
          queryParams.append('rand', 'true');
        }

        if (queryParams.toString()) {
          url += `?${queryParams.toString()}`;
        }

        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Request failed: ${res.status}`);
        }
        const data = await res.json();

        if (productId) {
          setSelectedProduct(data as Product);
          setProducts([data as Product]);
        } else if (search) {
          // Vector search returns a different structure
          const productList = data.map((item: any) => ({
            ...item,
            _id: item._id.toString(),
          }));
          setProducts(productList);
          setSelectedProduct(null);
        } else {
          const list = Array.isArray(data) ? data : (data && data.products) || [];
          setProducts(list);
          setSelectedProduct(null);
        }
      } catch (err) {
        console.error('Error fetching products:', err);
        setProducts([]);
        setProductError(String(err));
      } finally {
        setProductLoading(false);
      }
    };

    fetchProducts();
  }, [productId, search]);

  // Automatically load recommendations when a product is selected
  useEffect(() => {
    if (!selectedProduct) return;

    const fetchAllRecommendations = async () => {
      setRecLoading(true);
      setRecError(null);
      setWeightedHybridRecs([]);
      setRecLatencyMs(null);
      const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();

      try {
        // Fetch all recommendation types in parallel
        type RecommendationResponse = { recommendations: any[] };

        const [
          pca,
          review,
          contentPca,
          svd,
          weighted
        ]: RecommendationResponse[] = await Promise.all([
          getRecommendationsPCA(selectedProduct.product_id, 5),
          getRecommendationsReviews(selectedProduct.product_id, 5),
          getRecommendationsContentPCA(selectedProduct.product_id, 5),
          getRecommendationsSVD(selectedProduct.product_id, 5),
          getRecommendationsWeightedHybrid(selectedProduct.product_id, 5, {})
        ]);

        const allRecs = {
          pca_recs: pca?.recommendations || [],
          review_recs: review?.recommendations || [],
          content_pca_recs: contentPca?.recommendations || [],
          svd_recs: svd?.recommendations || [],
          recommendations: weighted?.recommendations || []
        };

        // Only set the weighted hybrid recs for display on this page
        setWeightedHybridRecs(allRecs.recommendations);

      } catch (err) {
        setRecError(String(err));
        setWeightedHybridRecs([]);
      } finally {
        const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
        setRecLatencyMs(Math.max(0, t1 - t0));
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

  if (search && !productId) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16 px-4">
        <div className="w-full max-w-5xl mb-4">
          <Link href="/" className="text-blue-500 hover:underline">
            &larr; Back to Home
          </Link>
        </div>
        <h1 className="text-3xl font-bold mb-8 text-black dark:text-zinc-50">
          Search Results for &quot;{search}&quot;
        </h1>
        {productLoading ? (
          <p>Loading...</p>
        ) : productError ? (
          <p className="text-red-500">Error: {productError}</p>
        ) : (
          <ProductGrid products={products} />
        )}
      </div>
    );
  }

  return (
    <div className="container mx-auto p-4">
      <Link href="/" className="text-blue-500 hover:underline mb-4 block">&larr; Back to Home</Link>
      {productLoading && <p>Loading products...</p>}
      {productError && <p className="text-red-500">Error: {productError}</p>}
      
      {/* Display Search Results */}
      <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16 px-4">
        <h1 className="text-3xl font-bold mb-8 text-black dark:text-zinc-50">
          {selectedProduct ? 'Product Details' : 'Products'}
        </h1>
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
          <div className="w-full max-w-5xl">
            <ProductGrid products={products} />
          </div>
        )}

        {selectedProduct && recLoading ? (
          <div className="w-full max-w-2xl mt-8 flex justify-center">
            <div className="flex items-center gap-3">
              <svg className="animate-spin h-6 w-6 text-gray-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"></path>
              </svg>
              <span className="text-gray-600">You may also like...</span>
            </div>
          </div>
        ) : selectedProduct && recError ? (
          <div className="w-full max-w-2xl mt-8 text-red-600">Error loading recommendations: {recError}</div>
        ) : selectedProduct && (
          <div className="w-full max-w-5xl mt-8 flex flex-col gap-8">
            <div>
              <h3 className="text-xl font-semibold mb-4">
                You may also like
                {recLatencyMs !== null && (
                  <span className="text-sm text-gray-500 ml-2"> (loaded in {Math.round(recLatencyMs)} ms)</span>
                )}
              </h3>
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