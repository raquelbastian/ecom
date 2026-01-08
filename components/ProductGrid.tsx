'use client';

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

interface ProductGridProps {
  products: Product[];
}

export default function ProductGrid({ products }: ProductGridProps) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
      {products.map((product) => (
        <div key={product.product_id} className="border rounded-lg p-4 flex flex-col items-center text-center h-full">
          <Link href={`/product-display?product_id=${encodeURIComponent(product.product_id)}`}>
            <div className="relative w-40 h-40">
              {product.img_link ? (
                <Image src={product.img_link} alt={product.product_name} layout="fill" objectFit="contain" className="rounded-t-lg" unoptimized />
              ) : <div className="w-full h-full bg-gray-200 rounded-t-lg"/>}
            </div>
          </Link>
          <div className="mt-2 flex-grow flex flex-col justify-between">
            <h4 className="font-semibold text-sm h-12 overflow-hidden">
              <Link href={`/product-display?product_id=${encodeURIComponent(product.product_id)}`}>{product.product_name}</Link>
            </h4>
            <div>
              <p className="mt-1 font-bold">${product.discounted_price}</p>
              <p className="mt-1 text-xs text-gray-500">Rating: {product.rating ?? 'N/A'}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
