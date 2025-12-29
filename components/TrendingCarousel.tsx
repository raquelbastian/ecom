"use client";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import React from "react";
import Link from "next/link";

interface Product {
  product_id: string;
  product_name: string;
  discounted_price: number | string;
  img_link?: string;
}

interface TrendingCarouselProps {
  trendingProducts: Product[];
}

const carouselSettings = {
  dots: true,
  infinite: true,
  speed: 500,
  slidesToShow: 3,
  slidesToScroll: 1,
  autoplay: true,
  autoplaySpeed: 2500,
  responsive: [
    {
      breakpoint: 1024,
      settings: {
        slidesToShow: 2,
      },
    },
    {
      breakpoint: 600,
      settings: {
        slidesToShow: 1,
      },
    },
  ],
};

export default function TrendingCarousel({ trendingProducts }: TrendingCarouselProps) {
  // Remove duplicates by product_id
  const uniqueProducts = React.useMemo(() => {
    const seen = new Set<string>();
    return trendingProducts.filter((p) => {
      if (seen.has(p.product_id)) return false;
      seen.add(p.product_id);
      return true;
    });
  }, [trendingProducts]);

  if (!uniqueProducts || uniqueProducts.length === 0) {
    return <p>No trending products found.</p>;
  }
  return (
    <Slider {...carouselSettings}>
      {uniqueProducts.map((product) => (
        <div key={product.product_id} style={{ padding: 16, height: 320 }}>
          <Link href={`/product-display?product_id=${product.product_id}`} style={{ textDecoration: 'none', color: 'inherit' }}>
            <div
              style={{
                border: '1px solid #eee',
                borderRadius: 12,
                padding: 16,
                background: '#fff',
                textAlign: 'center',
                boxShadow: '0 2px 8px rgba(0,0,0,0.05)',
                transition: 'box-shadow 0.2s',
                cursor: 'pointer',
                height: 288,
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'space-between',
                minWidth: 0,
                maxWidth: 320,
                margin: '0 auto',
              }}
            >
              {product.img_link && (
                <img
                  src={product.img_link}
                  alt={product.product_name}
                  style={{
                    width: '100%',
                    height: 140,
                    objectFit: 'contain',
                    marginBottom: 12,
                    borderRadius: 8,
                    background: '#fafafa',
                  }}
                />
              )}
              <h3 style={{ fontSize: 18, margin: '8px 0', minHeight: 48, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'normal', lineHeight: 1.2 }}>{product.product_name}</h3>
              <p style={{ fontWeight: 600, color: '#0070f3', margin: 0, fontSize: 20 }}>
                ${product.discounted_price}
              </p>
            </div>
          </Link>
        </div>
      ))}
    </Slider>
  );
}
