import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: false,
  },
  images: {
    unoptimized: true,
  },
  experimental: {
    serverActions: {
      bodySizeLimit: '15mb'
    }
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  
  // Webpack config for better performance
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push('canvas', 'jsdom');
    }
    return config;
  },
};

export default nextConfig;