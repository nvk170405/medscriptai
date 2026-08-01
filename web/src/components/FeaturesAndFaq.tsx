import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Scan, Shield, Zap, Brain, Users, GitBranch,
  ChevronDown, Plus, Minus
} from 'lucide-react';

// ── Features Section ─────────────────────────────────────────────────────────

const features = [
  {
    icon: Scan,
    title: 'Vision-First OCR',
    desc: 'Swin Transformer encoder processes cursive medical handwriting with sub-pixel attention across Indian prescription formats.',
  },
  {
    icon: Brain,
    title: 'Medical NER Pipeline',
    desc: 'BiomedBERT extracts medicines, dosages, frequencies, and instructions with domain-specific entity recognition.',
  },
  {
    icon: Zap,
    title: 'Real-Time Inference',
    desc: 'Sub-3 second transcription pipeline from image upload to structured JSON output, optimized for edge deployment.',
  },
  {
    icon: Shield,
    title: 'HIPAA-First Privacy',
    desc: 'JWT OAuth 2.0, role-based access control, and full audit logging ensure patient data stays protected.',
  },
  {
    icon: Users,
    title: 'Human-in-the-Loop',
    desc: 'Clinician corrections feed back into the training pipeline, continuously improving model accuracy over time.',
  },
  {
    icon: GitBranch,
    title: 'EHR Integration Ready',
    desc: 'Structured JSON outputs with ICD-10 codes map directly to electronic health record systems via REST API.',
  },
];

const FeatureCard = ({ feature, index }: { feature: typeof features[0]; index: number }) => {
  const Icon = feature.icon;
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.08 }}
      className="glass-panel rounded-2xl p-6 group hover:bg-white/[0.04] transition-colors"
    >
      <div className="w-10 h-10 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-4 group-hover:border-glowLight/30 transition-colors">
        <Icon className="w-5 h-5 text-glowLight" />
      </div>
      <h3 className="text-white font-medium mb-2">{feature.title}</h3>
      <p className="text-muted text-sm leading-relaxed">{feature.desc}</p>
    </motion.div>
  );
};

// ── FAQ Section ──────────────────────────────────────────────────────────────

const faqs = [
  {
    q: 'What types of prescriptions can MedScript process?',
    a: 'MedScript is trained on Indian medical handwriting styles including cursive prescriptions, clinical notes, and discharge summaries. It handles English and transliterated Hindi text, supporting a wide range of doctor handwriting patterns.',
  },
  {
    q: 'Does the system require an internet connection?',
    a: 'The model is designed for edge/cloud hybrid deployment. The core inference pipeline (Swin Transformer + BiLSTM) runs locally without internet. Cloud connectivity is only needed for model updates and the feedback loop.',
  },
  {
    q: 'How is patient data protected?',
    a: 'All data is processed with HIPAA-compliant security: JWT-based authentication, role-based access control (RBAC), full audit logging, and encrypted storage. Images are processed in-memory and never persisted to disk unless explicitly configured.',
  },
  {
    q: 'How accurate is the entity extraction?',
    a: 'On our benchmark dataset, the pipeline achieves 98.2% entity extraction accuracy for medicine names, dosages, and frequencies. The human-in-the-loop feedback system continuously improves accuracy by retraining on clinician-verified corrections.',
  },
  {
    q: 'Can I integrate MedScript with existing hospital systems?',
    a: 'Yes. The REST API outputs structured JSON with extracted entities, making it straightforward to integrate with EHR systems, pharmacy management software, and clinical analytics platforms.',
  },
];

const FaqItem = ({ faq, index }: { faq: typeof faqs[0]; index: number }) => {
  const [open, setOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.05 }}
      className="border-b border-white/5 last:border-none"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between py-5 text-left group"
      >
        <span className="text-white text-sm font-medium pr-4 group-hover:text-glowLight transition-colors">
          {faq.q}
        </span>
        <div className="w-6 h-6 rounded-full bg-white/5 border border-white/10 flex items-center justify-center flex-shrink-0">
          {open ? (
            <Minus className="w-3 h-3 text-glowLight" />
          ) : (
            <Plus className="w-3 h-3 text-muted" />
          )}
        </div>
      </button>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <p className="text-muted text-sm leading-relaxed pb-5 pr-12">
              {faq.a}
            </p>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// ── Combined Export ──────────────────────────────────────────────────────────

export const FeaturesAndFaq = () => {
  return (
    <>
      {/* Features */}
      <section className="relative z-10 max-w-6xl mx-auto px-4 py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-semibold gradient-text mb-4">
            Built for Clinical Precision
          </h2>
          <p className="text-muted text-lg max-w-xl mx-auto">
            Every component is purpose-built for the unique challenges of medical handwriting digitization.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {features.map((f, i) => (
            <FeatureCard key={f.title} feature={f} index={i} />
          ))}
        </div>
      </section>

      {/* Pipeline Visual */}
      <section className="relative z-10 max-w-4xl mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-panel rounded-3xl p-8 md:p-12"
        >
          <h3 className="text-white font-semibold text-xl mb-8 text-center">Inference Pipeline</h3>
          <div className="flex flex-col md:flex-row items-center gap-4 md:gap-0">
            {[
              { step: '01', label: 'Image Input', sub: 'OpenCV preprocessing' },
              { step: '02', label: 'Vision Encoder', sub: 'Swin Transformer' },
              { step: '03', label: 'Sequence Decode', sub: 'BiLSTM + CTC' },
              { step: '04', label: 'Entity Extract', sub: 'BiomedBERT NER' },
              { step: '05', label: 'Structured JSON', sub: 'EHR-ready output' },
            ].map((s, i) => (
              <React.Fragment key={s.step}>
                <div className="flex-1 text-center">
                  <div className="text-xs text-glowLight font-mono mb-1">{s.step}</div>
                  <div className="text-white text-sm font-medium">{s.label}</div>
                  <div className="text-muted text-xs mt-0.5">{s.sub}</div>
                </div>
                {i < 4 && (
                  <div className="hidden md:block w-8 h-[1px] bg-white/10 flex-shrink-0">
                    <div className="w-2 h-2 border-r border-t border-white/20 rotate-45 -mt-[3px] ml-auto" />
                  </div>
                )}
              </React.Fragment>
            ))}
          </div>
        </motion.div>
      </section>

      {/* FAQ */}
      <section className="relative z-10 max-w-3xl mx-auto px-4 py-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-semibold gradient-text mb-4">
            Frequently Asked
          </h2>
          <p className="text-muted text-lg">
            Common questions about the platform.
          </p>
        </motion.div>

        <div className="glass-panel rounded-3xl px-6 md:px-8">
          {faqs.map((f, i) => (
            <FaqItem key={i} faq={f} index={i} />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-12 px-4">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 bg-white rounded-full flex items-center justify-center overflow-hidden">
              <div className="w-2.5 h-2.5 bg-background rounded-full translate-x-[1.5px]"></div>
            </div>
            <span className="text-sm text-muted">MedScript AI</span>
          </div>
          <p className="text-xs text-muted/50">
            © 2026 MedScript AI · Privacy-first clinical intelligence · Built by Navketan Singh
          </p>
        </div>
      </footer>
    </>
  );
};
