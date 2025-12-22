import CategorySlot from '@/components/CategorySlot';

const CATEGORIES = [
  'Computers & Accessories',
  'Electronics',
  'Home & Kitchen',
];
const ITEMS_PER_SLOT = 12;

async function fetchProductsForCategory(category: string) {
  try {
    // Build an absolute URL for server-side fetch. Use NEXT_PUBLIC_BASE_URL if provided,
    // otherwise fall back to localhost:3000 which works in local development.
    const origin = process.env.NEXT_PUBLIC_BASE_URL || process.env.NEXT_PUBLIC_SITE_URL || 'http://localhost:3000';
    const url = new URL('/api/products', origin);
    url.searchParams.set('category', category);
    url.searchParams.set('limit', String(ITEMS_PER_SLOT));
    // Add a random param to force different results on each load
    url.searchParams.set('rand', Math.random().toString());

    const res = await fetch(url.toString(), { cache: 'no-store' });
    if (!res.ok) {
      console.error('Failed to fetch products for', category, res.status);
      return [];
    }
    const data = await res.json();
    // API may return an array or an object with { products: [...] }
    if (Array.isArray(data)) return data;
    if (data && Array.isArray(data.products)) return data.products;
    return [];
  } catch (err) {
    console.error('Error fetching products for category', category, err);
    return [];
  }
}

export default async function Home() {
  // Fetch each category's top items in parallel on the server
  const fetches = CATEGORIES.map(c => fetchProductsForCategory(c));
  const results = await Promise.all(fetches);

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">
      <div className="w-full max-w-5xl px-6">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-black dark:text-zinc-50">Welcome</h1>
          <p className="text-zinc-600 dark:text-zinc-400">Browse curated slots powered by server-rendered data.</p>
        </header>

        <section className="space-y-12">
          {CATEGORIES.map((cat, idx) => (
            <CategorySlot key={cat} title={cat} products={results[idx] || []} />
          ))}
        </section>
      </div>
    </div>
  );
}
