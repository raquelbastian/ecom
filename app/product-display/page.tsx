import { Suspense } from 'react';
import ProductDisplayClient from './ProductDisplayClient';

export default function ProductDisplayPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center py-16">Loading...</div>}>
      <ProductDisplayClient />
    </Suspense>
  );
}
