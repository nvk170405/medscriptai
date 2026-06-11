import React from 'react';
import { motion } from 'framer-motion';

const BentoCard = ({ children, className = "" }: any) => (
  <div className={`glass-panel rounded-3xl p-6 ${className}`}>
    {children}
  </div>
);

export const BentoDashboard = () => {
  return (
    <section className="relative z-10 max-w-7xl mx-auto px-4 py-32 flex flex-col items-center">
      
      <div className="text-center mb-16">
        <motion.h2 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-4xl md:text-5xl font-semibold gradient-text mb-4"
        >
          Meet Marvellous Insights
        </motion.h2>
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="text-muted text-lg"
        >
          Save your clinical team's precious time. MedScript replaces manual EHR data entry.
        </motion.p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 w-full max-w-5xl">
        
        {/* Top Left - Large Metric */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="lg:col-span-2"
        >
          <BentoCard className="h-full flex flex-col justify-between min-h-[300px]">
            <div>
              <div className="text-6xl font-semibold text-white mb-2">98.2%</div>
              <div className="text-muted">Entity Extraction Accuracy</div>
            </div>
            
            {/* Minimalist World Map placeholder (using dots) */}
            <div className="absolute right-12 top-12 opacity-30 pointer-events-none">
              <svg width="200" height="100" viewBox="0 0 200 100">
                <circle cx="50" cy="40" r="2" fill="white" />
                <circle cx="80" cy="30" r="1.5" fill="white" />
                <circle cx="120" cy="50" r="2.5" fill="white" />
                <circle cx="150" cy="40" r="1" fill="white" />
                <path d="M50 40 Q 80 10 120 50 T 150 40" fill="none" stroke="white" strokeWidth="0.5" strokeDasharray="4 4" />
              </svg>
            </div>

            <div className="mt-8 pt-6 border-t border-white/5 flex gap-4">
              <div className="flex-1">
                <h4 className="text-white font-medium mb-1">Success Transactions</h4>
                <p className="text-xs text-muted">Innovative vision technology meets clinical expertise.</p>
              </div>
            </div>
          </BentoCard>
        </motion.div>

        {/* Top Right - Bar Chart */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
        >
          <BentoCard className="h-full flex flex-col min-h-[300px]">
            <div className="flex items-end gap-2 h-40 mt-4 mb-8">
              {[40, 65, 45, 85, 55, 30].map((h, i) => (
                <motion.div 
                  key={i}
                  initial={{ height: 0 }}
                  whileInView={{ height: `${h}%` }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.4 + i * 0.1, duration: 0.8, type: "spring" }}
                  className="w-full bg-gradient-to-t from-glowLight/20 to-white/40 rounded-t-sm"
                />
              ))}
            </div>
            <div>
              <h4 className="text-white font-medium mb-1">Processing Labyrinth</h4>
              <p className="text-xs text-muted">Where each stroke is a smart token</p>
            </div>
          </BentoCard>
        </motion.div>

        {/* Bottom Left - Small Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.1 }}
          className="flex flex-col gap-6"
        >
          <div className="flex gap-6">
            <BentoCard className="flex-1 flex flex-col items-start px-5 py-6">
              <div className="w-1 h-6 bg-glowLight rounded-full mb-3"></div>
              <div className="text-xs text-muted mb-1">Financial</div>
              <div className="text-xl font-semibold mb-1">19.2<span className="text-xs font-normal text-muted ml-1">k</span></div>
              <div className="text-xs text-glowLight">Prescriptions</div>
            </BentoCard>
            <BentoCard className="flex-1 flex flex-col items-start px-5 py-6">
               <div className="w-1 h-6 bg-white rounded-full mb-3"></div>
              <div className="text-xs text-muted mb-1">Growth</div>
              <div className="text-xl font-semibold mb-1">24<span className="text-xs font-normal text-muted ml-1">%</span></div>
              <div className="text-xs text-white/50">Efficiency</div>
            </BentoCard>
          </div>
          <BentoCard className="flex-1 flex flex-col justify-center">
            <h4 className="text-white font-medium mb-1">Your Clinical Palette</h4>
            <p className="text-xs text-muted">Watch your data grow in a thriving ecosystem so easily</p>
          </BentoCard>
        </motion.div>

        {/* Bottom Right - Large Chart */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.3 }}
          className="lg:col-span-2"
        >
          <BentoCard className="h-full flex flex-col md:flex-row gap-8">
            <div className="flex-1">
              <h4 className="text-white font-medium mb-2">MedScript Space Opportunities</h4>
              <p className="text-xs text-muted leading-relaxed">
                Where each transcription is a smart contract and every color is a chance to build a comprehensive patient history.
              </p>
            </div>
            <div className="flex-1 flex items-end gap-3 h-32 md:h-auto">
               {[
                 { h: '60%', c: 'bg-gradient-to-t from-red-500/20 to-red-400/80' },
                 { h: '40%', c: 'bg-gradient-to-t from-orange-500/20 to-orange-400/80' },
                 { h: '90%', c: 'bg-gradient-to-t from-white/20 to-white/80' },
                 { h: '50%', c: 'bg-gradient-to-t from-glowLight/20 to-glowLight/80' },
                 { h: '70%', c: 'bg-gradient-to-t from-green-500/20 to-green-400/80' },
               ].map((bar, i) => (
                <div key={i} className="flex-1 flex flex-col items-center justify-end h-full">
                  <div className="text-[10px] text-muted mb-2">{Math.floor(Math.random() * 50 + 10)}</div>
                  <motion.div 
                    initial={{ height: 0 }}
                    whileInView={{ height: bar.h }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.6 + i * 0.1, duration: 1, type: "spring" }}
                    className={`w-full ${bar.c} rounded-t-sm`}
                  />
                </div>
               ))}
            </div>
          </BentoCard>
        </motion.div>

      </div>
    </section>
  );
};
