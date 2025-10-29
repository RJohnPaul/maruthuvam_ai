import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Button } from './components/ui/button';
import { MainMenusGradientCard } from "@/cuicui/other/cursors/dynamic-cards/gradient-card";
import {
  FileText,
  Clock,
  Brain,
  CloudLightning,
  Shield,
  Database,
  ArrowRight,
  ChevronDownCircleIcon
} from 'lucide-react';
import { HeroList } from './components/HeroList';
import { InteractiveHoverButton } from "./components/magicui/interactive-hover-button";
import { RippleButton } from "./components/magicui/ripple-button";
import { Ripple } from "./components/magicui/ripple";
import { SparklesText } from "./components/magicui/sparkles-text";
import { Feedback } from './components/FeedBackCard';
import Testimonials from './components/Testimonials';
import { ScratchToReveal } from "./components/magicui/scratch-to-reveal";
import FaqSection from './components/FaqSection';
import PrivacySection from './components/PrivacySection';
import { AuroraText } from "./components/magicui/aurora-text";
import { MagicCard } from "./components/magicui/magic-card";

const LandingPage = () => {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 50);
    return () => clearTimeout(t);
  }, []);

  const cardBase =
    "relative overflow-hidden rounded-2xl border backdrop-blur-md bg-white/10 dark:bg-white/5 border-white/20 dark:border-white/10 shadow-lg hover:shadow-xl transition-all";
  const cardAnim = (i) =>
    `${mounted ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'} transition-all duration-700 ease-out` +
    ` `;
  const delayStyle = (i) => ({ transitionDelay: `${i * 120}ms` });

  return (
    <div className="flex flex-col w-full">
      {/* <SmoothCursor /> */}
      {/* Hero Section */}
      <section className="relative text-black dark:text-white py-16 md:py-24 overflow-hidden">
        <div className="container mx-auto px-4 md:px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <div className="inline-flex items-center rounded-full border border-slate-700 bg-white dark:bg-slate-800/50 px-3 py-1 text-sm text-black dark:text-slate-300">
                <span className="flex h-2 w-2 rounded-full bg-blue-400 mr-2"></span>
                Medical Imaging Enhanced by AI
              </div>
              <h1 className="text-4xl md:text-5xl lg:text-6xl tracking-tight">
                Diagnose Smarter, <br />
                <SparklesText>
                  <text className='text-[#4520d7] drop-shadow-[0_0_6px_#2400b8] font-semibold'>Faster, Better</text>
                </SparklesText>
              </h1>

              <p className="text-gray-600 dark:text-slate-300 text-lg md:text-xl max-w-md">
                Harness the power of 5 specialized AI models to analyze medical images with unprecedented accuracy and speed.
              </p>

              <div className="pt-4 flex flex-wrap gap-4">

                <Link to="/upload">
                  <InteractiveHoverButton>Start Diagnosis</InteractiveHoverButton>
                </Link>


                <a href="#features">
                  <RippleButton className='rounded-full'><span className='flex items-center justify-center gap-2 font-smeibold'>Learn More <ChevronDownCircleIcon className='text-gray-500' /></span></RippleButton>
                </a>

              </div>

              <div className="pt-6 flex items-center text-black dark:text-slate-400 text-sm">
                <div className="flex -space-x-2 mr-3">
                  <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center text-xs font-medium text-white">JD</div>
                  <div className="h-8 w-8 rounded-full bg-green-600 flex items-center justify-center text-xs font-medium text-white">SL</div>
                  <div className="h-8 w-8 rounded-full bg-amber-600 flex items-center justify-center text-xs font-medium text-white">RK</div>
                </div>
                Trusted by 5,000+ medical professionals worldwide
              </div>
            </div>

            <div className="hidden md:block relative">
              {/* <div className="absolute -left-8 -top-8 w-64 h-64 bg-blue-500 rounded-full filter blur-3xl opacity-20"></div>
              <div className="absolute -right-8 -bottom-8 w-64 h-64 bg-purple-500 rounded-full filter blur-3xl opacity-20"></div> */}

              <HeroList />

            </div>
          </div>
        </div>
        {/* Progressive blur + gradient to smooth into next section */}
        <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24 z-10">
          <div className="h-full w-full bg-gradient-to-b from-transparent via-white/70 to-white dark:via-zinc-900/60 dark:to-zinc-950 backdrop-blur-md" />
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative -mt-10 pt-16 pb-16 bg-white dark:bg-zinc-950">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight mb-2">Advanced Features</h2>
            <p className="text-slate-600 max-w-2xl mx-auto">
              MaruthuvamAI combines cutting-edge technology with medical expertise to deliver unparalleled diagnostic assistance.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {/* Feature 1 */}
            <div className={`${cardAnim(0)}`} style={delayStyle(0)}>
              <MainMenusGradientCard
                title="5 Specialized AI Models"
                description="Purpose-built AI models optimized for specific medical imaging modalities, from brain MRIs to retinal scans."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <Brain className="h-6 w-6 text-blue-600" />
                </div>
              </MainMenusGradientCard>
            </div>

            {/* Feature 2 */}
            <div className={`${cardAnim(1)}`} style={delayStyle(1)}>
              <MainMenusGradientCard
                title="LLM-Based Reporting"
                description="Natural language reports generated instantly from image analysis, saving hours of documentation time."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <CloudLightning className="h-6 w-6 text-green-600" />
                </div>
              </MainMenusGradientCard>
            </div>

            {/* Feature 3 */}
            <div className={`${cardAnim(2)}`} style={delayStyle(2)}>
              <MainMenusGradientCard
                title="Instant PDF Reports"
                description="One-click generation of comprehensive, shareable reports with segmentation visualization."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <FileText className="h-6 w-6 text-amber-600" />
                </div>
              </MainMenusGradientCard>
            </div>

            {/* Feature 4 */}
            <div className={`${cardAnim(3)}`} style={delayStyle(3)}>
              <MainMenusGradientCard
                title="Rapid Processing"
                description="Analysis completed in seconds, not hours, enabling faster clinical decision-making."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <Clock className="h-6 w-6 text-purple-600" />
                </div>
              </MainMenusGradientCard>
            </div>

            {/* Feature 5 */}
            <div className={`${cardAnim(4)}`} style={delayStyle(4)}>
              <MainMenusGradientCard
                title="HIPAA Compliant"
                description="Enterprise-grade security with full encryption and compliance with medical data regulations."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <Shield className="h-6 w-6 text-red-600" />
                </div>
              </MainMenusGradientCard>
            </div>

            {/* Feature 6 */}
            <div className={`${cardAnim(5)}`} style={delayStyle(5)}>
              <MainMenusGradientCard
                title="Historical Analysis"
                description="Compare current results with patient history to identify changes and trends over time."
                withArrow
              >
                <div className="rounded-lg bg-white/20 dark:bg-white/10 p-3 w-12 h-12 flex items-center justify-center">
                  <Database className="h-6 w-6 text-indigo-600" />
                </div>
              </MainMenusGradientCard>
            </div>
          </div>
        </div>
      </section>

      {/* security claim */}
      <section className='mx-auto py-16 w-full px-2 md:px-44 bg-white dark:bg-zinc-950'>
        <MagicCard className='relative overflow-hidden rounded-2xl bg-transparent text-black dark:text-white shadow-xl'>
          <PrivacySection/>
        </MagicCard>
      </section>

      {/* CTA Section */}
      <section className=" bg-white dark:bg-zinc-950">
        <div className="relative flex h-[500px] w-full flex-col items-center justify-center overflow-hidden rounded-lg bg-background">
          <div className="container mx-auto px-4 md:px-6">
            <div className="max-w-3xl mx-auto text-center">
              <h2 className="text-5xl font-bold tracking-tight mb-4">
                Ready to Transform Your Diagnostic Workflow?
              </h2>
              <p className="text-slate-600 mb-8 max-w-xl mx-auto">
                Join thousands of medical professionals already using MaruthuvamAI to improve accuracy and save time.
              </p>
              <Button
                asChild
                size="lg"
                className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white px-8 shadow-lg rounded-full transition-transform transform hover:scale-105"
              >
                <Link to="/upload">
                
                  Start Your First Analysis <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
          <Ripple />
        </div>

      </section>


      {/* Contact Section */}
      <section className="py-16 px-4 md:px-96 flex items-center justify-between bg-white dark:bg-zinc-950 transition-colors gap-2">
        <div className="container">
          <div className="mb-12">
            <h2 className="text-3xl font-bold tracking-tight mb-2 text-zinc-900 dark:text-white">
              Get in Touch
            </h2>
            <p className="text-slate-600 dark:text-slate-400">
              Have questions or need support? Our team is here to help you.
            </p>
          </div>

          <div className="flex">
            <Button
              asChild
              size="lg"
              className="bg-white text-black px-8 shadow-lg rounded-full transition-transform transform hover:scale-105 text-sm font-bold"
            >
              <Link to="/contact">Contact Us</Link>
            </Button>
          </div>
        </div>
        <div>
          <ScratchToReveal
      width={250}
      height={250}
      minScratchPercentage={70}
      className="flex items-center justify-center overflow-hidden rounded-2xl border-2 bg-gray-100"
      gradientColors={["#A97CF8", "#F38CB8", "#FDCC92"]}
    >
      <p className="text-9xl">💉</p>
    </ScratchToReveal>
        </div>
      </section>


      {/* Feedback Section */}
      <>
        <div className="bg-white dark:bg-zinc-950 transition-colors">
          <div className="container mx-auto px-4 md:px-6">
            <div className="text-center mb-12">
              <h2 className="text-3xl font-bold tracking-tight mb-2">What Our Users Say</h2>
              <p className="text-slate-600 max-w-2xl mx-auto">
                Hear from our satisfied users about how MaruthuvamAI has transformed their diagnostic processes.
              </p>
            </div>
          </div>
        </div>
        <div >
          <Testimonials />
        </div>
        <div className="flex items-center justify-center bg-white dark:bg-zinc-950 py-5">
          <Feedback />
        </div>
      </>

      {/* FAQ Section */}
      <section className="py-16 bg-white dark:bg-zinc-950">
        <div className="container mx-auto px-4 md:px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold tracking-tight mb-2">Frequently Asked Questions</h2>
            <p className="text-slate-600 max-w-2xl mx-auto">
              Have questions? We have answers. Check out our FAQ section for more information.
            </p>
          </div>
          {/* Add your FAQ component here */}
          <FaqSection/>
        </div>
      </section>

    </div>
  );
};

export default LandingPage;