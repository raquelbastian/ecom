"use client";

import Image from 'next/image';
import Link from 'next/link';

interface Props {
  title: string;
  products: any[];
}

export default function CategorySlot({ title, products }: Props) {
  return (
    <section>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-semibold text-black dark:text-zinc-50">{title}</h2>
        <Link href={`/?category=${encodeURIComponent(title)}`} className="text-sm text-blue-600">See all</Link>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
        {products.length === 0 && (
          <div className="text-zinc-500">No items</div>
        )}

        {products.map((p: any) => (
          <Link key={p.product_id} href={`/product-display?product_id=${encodeURIComponent(p.product_id)}`} className="block border rounded-lg bg-white dark:bg-zinc-900 p-3 hover:shadow">
            <div className="relative w-full" style={{ paddingTop: '56.25%' }}>
              {p.img_link ? (
                <>
                  <Image src={p.img_link} alt={p.product_name || ''} fill style={{ objectFit: 'cover' }} unoptimized />
                  <img src={p.img_link} alt="" style={{ display: 'none' }} onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = 'none'; }} />
                </>
              ) : (
                <div className="absolute inset-0 w-full h-full bg-zinc-100 dark:bg-zinc-800"></div>
              )}
            </div>
            <h3 className="mt-2 text-sm font-medium text-black dark:text-zinc-50 truncate">{p.product_name}</h3>
            <div className="mt-1 text-sm text-zinc-700 dark:text-zinc-300">${p.discounted_price}</div>
          </Link>
        ))}
      </div>
    </section>
  );
}
