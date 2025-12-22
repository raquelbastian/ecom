import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    // Allow external images from any host for development. Restrict this in production.
    remotePatterns: [
      { protocol: 'https', hostname: '**' },
      { protocol: 'http', hostname: '**' },
    ],
  },
};

export default nextConfig;
