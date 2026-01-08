import CategorySlot from '@/components/CategorySlot';
import { getTrendingProductsML } from '@/lib/mlService';
import TrendingCarousel from "@/components/TrendingCarousel";
import { getProducts } from '@/lib/mongodb';
import SearchBar from '@/components/SearchBar';

const CATEGORIES = [
  'Computers & Accessories',
  'Electronics',
  'Home & Kitchen',
];
const ITEMS_PER_SLOT = 12;

export default async function HomePage() {
  // Fetch each category's top items in parallel on the server
  const fetches = CATEGORIES.map(c => getProducts({ category: c, limit: ITEMS_PER_SLOT, randomize: true }));
  const results = await Promise.all(fetches);
  console.log('Products fetched for homepage:', results);

  // Fetch ML-based trending products (server-side)
  let trendingProducts: any[] = [];
  try {
    const trendingRes = await getTrendingProductsML(8);
    trendingProducts = trendingRes?.trending || [];
  } catch (err) {
    trendingProducts = [];
  }

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">
      <div className="w-full max-w-5xl px-6">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-black dark:text-zinc-50">Raq's eCommerce store</h1>
        </header>

        <div className="my-8 w-full">
          <SearchBar />
        </div>

        {/* Trending Products Section */}
        <section className="mb-12">
          <h2 className="text-2xl font-semibold text-yellow-500 dark:text-yellow-500 mb-4">Trending Products</h2>
          <TrendingCarousel trendingProducts={trendingProducts} />
        </section>

        <section className="space-y-12">
          {CATEGORIES.map((cat, idx) => {
            const categoryProducts = results[idx];
            const productsArray = Array.isArray(categoryProducts) ? categoryProducts : [];
            return <CategorySlot key={cat} title={cat} products={productsArray} />;
          })}
        </section>
      </div>
    </main>
  );
}
