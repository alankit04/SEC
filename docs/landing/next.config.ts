import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",          // static export — drop dist/ into any CDN
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
