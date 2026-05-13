/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",          // static export — drop dist/ into any CDN
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
