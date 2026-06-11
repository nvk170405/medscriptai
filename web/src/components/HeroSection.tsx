import React, { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import gsap from 'gsap';
import { ArrowRight, Sparkles, ChevronDown } from 'lucide-react';

const FloatingNode = ({ label, value, icon, x, y, delay }: any) => {
  const nodeRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (nodeRef.current) {
      // Floating animation
      gsap.to(nodeRef.current, {
        y: "+=15",
        x: "+=10",
        duration: 3 + Math.random() * 2,
        repeat: -1,
        yoyo: true,
        ease: "sine.inOut",
        delay: delay
      });
    }
  }, [delay]);

  return (
    <div 
      ref={nodeRef} 
      className="absolute flex items-center gap-3 pointer-events-none"
      style={{ left: x, top: y }}
    >
      <div className="w-8 h-8 rounded-full bg-surface border border-white/10 flex items-center justify-center shadow-[0_0_15px_rgba(125,162,145,0.2)]">
        {icon}
      </div>
      <div>
        <div className="flex items-center gap-2">
          <div className="w-1 h-1 rounded-full bg-glowLight"></div>
          <span className="text-sm font-medium text-white">{label}</span>
        </div>
        <div className="text-xs text-muted pl-3">{value}</div>
      </div>
      
      {/* Connecting lines - simplified for the effect */}
      <svg className="absolute top-1/2 -left-12 w-12 h-[1px] opacity-20 -translate-y-1/2">
        <line x1="0" y1="0" x2="100%" y2="0" stroke="white" strokeWidth="1" strokeDasharray="2 2" />
      </svg>
    </div>
  );
};

export const HeroSection = () => {
  const linesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Vertical light trails animation
    if (linesRef.current) {
      const lines = linesRef.current.children;
      gsap.fromTo(lines, 
        { y: -100, opacity: 0 },
        {
          y: 400,
          opacity: 0,
          duration: 3,
          stagger: {
            each: 0.8,
            repeat: -1,
            amount: 2
          },
          ease: "power1.inOut"
        }
      );
    }
  }, []);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center pt-20 overflow-hidden z-10">
      
      {/* Vertical Data Trails */}
      <div ref={linesRef} className="absolute inset-0 flex justify-center gap-16 md:gap-32 pointer-events-none opacity-40">
        <div className="w-[1px] h-32 bg-gradient-to-b from-transparent via-white to-transparent shadow-[0_0_8px_rgba(255,255,255,0.8)]"></div>
        <div className="w-[1px] h-24 bg-gradient-to-b from-transparent via-glowLight to-transparent shadow-[0_0_8px_rgba(155,187,170,0.8)] mt-32"></div>
        <div className="w-[1px] h-40 bg-gradient-to-b from-transparent via-white to-transparent shadow-[0_0_8px_rgba(255,255,255,0.8)] mt-12"></div>
        <div className="w-[1px] h-20 bg-gradient-to-b from-transparent via-glowLight to-transparent shadow-[0_0_8px_rgba(155,187,170,0.8)] mt-48"></div>
      </div>

      {/* Floating Nodes */}
      <FloatingNode 
        label="BiomedBERT" value="NER 99.1%" 
        icon={<div className="w-2 h-2 border border-glowLight rotate-45"></div>}
        x="15%" y="30%" delay={0} 
      />
      <FloatingNode 
        label="Swin-Transformer" value="Vision 96.4%" 
        icon={<div className="w-3 h-3 border border-white rounded-sm"></div>}
        x="75%" y="25%" delay={1.2} 
      />
      <FloatingNode 
        label="BiLSTM + CTC" value="Decoder 98.2%" 
        icon={<div className="flex gap-0.5"><div className="w-1 h-2 bg-glowLight rounded-full"></div><div className="w-1 h-3 bg-glowLight rounded-full"></div><div className="w-1 h-1.5 bg-glowLight rounded-full"></div></div>}
        x="80%" y="65%" delay={0.7} 
      />
      <FloatingNode 
        label="OpenCV" value="Pre-processing" 
        icon={<div className="w-3 h-3 rounded-full border border-white border-dashed"></div>}
        x="10%" y="60%" delay={2.1} 
      />

      {/* Main Content */}
      <div className="relative z-20 flex flex-col items-center text-center max-w-4xl px-4">
        
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-md mb-8 cursor-pointer hover:bg-white/10 transition-colors"
        >
          <Sparkles className="w-4 h-4 text-glowLight" />
          <span className="text-sm text-gray-300">Unlock Clinical Intelligence</span>
          <ArrowRight className="w-4 h-4 text-gray-400 ml-2" />
        </motion.div>

        <motion.h1 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          className="text-5xl md:text-7xl font-semibold tracking-tight gradient-text mb-6 leading-tight"
        >
          Absolute Precision<br />for Medical Records
        </motion.h1>

        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
          className="text-lg text-muted max-w-2xl mb-12"
        >
          Dive into structured clinical data, where advanced computer vision meets state-of-the-art medical NLP to eradicate transcription errors.
        </motion.p>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6, ease: "easeOut" }}
          className="flex items-center gap-6"
        >
          <button className="px-6 py-3 rounded-full bg-white/5 border border-white/10 text-white hover:bg-white/10 transition-all flex items-center gap-2">
            Open App <ArrowRight className="w-4 h-4" />
          </button>
          <button className="px-6 py-3 rounded-full bg-white text-black font-medium hover:bg-gray-200 transition-all">
            Discover More
          </button>
        </motion.div>

      </div>

      {/* Scroll Indicator */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-12 left-12 flex items-center gap-3 text-sm text-muted"
      >
        <div className="w-8 h-8 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
          <ChevronDown className="w-4 h-4" />
        </div>
        <span>01/03 . Scroll down</span>
      </motion.div>

      {/* API Horizon Progress - Bottom Right */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-12 right-12 text-sm text-muted"
      >
        <div className="mb-2 text-glowLight">MedScript pipeline</div>
        <div className="flex gap-1">
          <div className="w-8 h-1 bg-white rounded-full"></div>
          <div className="w-8 h-1 bg-white/20 rounded-full"></div>
          <div className="w-8 h-1 bg-white/20 rounded-full"></div>
        </div>
      </motion.div>

    </section>
  );
};
