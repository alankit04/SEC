import Navbar      from "@/components/Navbar";
import Hero        from "@/components/Hero";
import Features    from "@/components/Features";
import HowItWorks  from "@/components/HowItWorks";
import Stats       from "@/components/Stats";
import Security    from "@/components/Security";
import CTA         from "@/components/CTA";
import Footer      from "@/components/Footer";

export default function LandingPage() {
  return (
    <main className="relative">
      <Navbar />
      <Hero />
      <Features />
      <HowItWorks />
      <Stats />
      <Security />
      <CTA />
      <Footer />
    </main>
  );
}
