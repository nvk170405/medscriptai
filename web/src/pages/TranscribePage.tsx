import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { transcribeImage, type TranscriptionResponse, type Entity } from '../lib/api';
import { Upload, FileImage, Loader2, CheckCircle2, AlertTriangle, Sparkles } from 'lucide-react';

const confidenceColor = (c: number) => {
  if (c >= 0.9) return 'text-emerald-400';
  if (c >= 0.7) return 'text-amber-400';
  return 'text-red-400';
};

const confidenceBg = (c: number) => {
  if (c >= 0.9) return 'bg-emerald-500';
  if (c >= 0.7) return 'bg-amber-500';
  return 'bg-red-500';
};

const entityIcon = (type: string) => {
  switch (type) {
    case 'medicine': return '💊';
    case 'dosage': return '⚖️';
    case 'frequency': return '🔁';
    case 'duration': return '📅';
    case 'instruction': return '📝';
    default: return '🏷️';
  }
};

export default function TranscribePage() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [result, setResult] = useState<TranscriptionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] },
    maxSize: 10 * 1024 * 1024,
    multiple: false,
  });

  const handleTranscribe = async () => {
    if (!file) return;
    setIsLoading(true);
    setError(null);

    try {
      const res = await transcribeImage(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Transcription failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen pt-28 pb-16 px-4 relative z-10">
      <div className="max-w-6xl mx-auto">

        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-semibold gradient-text mb-3">Transcribe Prescription</h1>
          <p className="text-muted">Upload a handwritten prescription and watch AI extract structured data in real-time.</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">

          {/* Left - Upload Area */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
          >
            <div className="glass-panel rounded-3xl p-6 h-full flex flex-col">
              <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                <FileImage className="w-5 h-5 text-glowLight" />
                Source Document
              </h3>

              {!file ? (
                <div
                  {...getRootProps()}
                  className={`flex-1 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-12 cursor-pointer transition-all min-h-[300px] ${
                    isDragActive
                      ? 'border-glowLight/60 bg-glowLight/5'
                      : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'
                  }`}
                >
                  <input {...getInputProps()} />
                  <Upload className="w-10 h-10 text-muted mb-4" />
                  <p className="text-white font-medium mb-1">Drag & drop your prescription here</p>
                  <p className="text-muted text-sm">or click to browse · JPG, PNG up to 10MB</p>
                </div>
              ) : (
                <div className="flex-1 flex flex-col">
                  <div className="flex-1 rounded-2xl overflow-hidden bg-black/20 border border-white/5">
                    <img src={preview!} alt="Prescription" className="w-full h-full object-contain" />
                  </div>
                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={handleTranscribe}
                      disabled={isLoading}
                      className="flex-1 flex items-center justify-center gap-2 bg-white text-black font-medium py-3 rounded-xl hover:bg-gray-200 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <Sparkles className="w-5 h-5" />
                      )}
                      {isLoading ? 'Processing...' : 'Transcribe'}
                    </button>
                    <button
                      onClick={handleReset}
                      className="px-4 py-3 rounded-xl bg-white/5 border border-white/10 text-muted hover:text-white hover:bg-white/10 transition-all"
                    >
                      Reset
                    </button>
                  </div>
                </div>
              )}
            </div>
          </motion.div>

          {/* Right - Results */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <div className="glass-panel rounded-3xl p-6 h-full flex flex-col min-h-[400px]">
              <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-glowLight" />
                Structured Output
              </h3>

              {error && (
                <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3 mb-4">
                  <AlertTriangle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}

              <AnimatePresence mode="wait">
                {!result && !isLoading && (
                  <motion.div
                    key="empty"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex-1 flex flex-col items-center justify-center text-muted"
                  >
                    <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-4">
                      <Upload className="w-8 h-8" />
                    </div>
                    <p className="text-sm">Upload a prescription to see results</p>
                  </motion.div>
                )}

                {isLoading && (
                  <motion.div
                    key="loading"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex-1 flex flex-col items-center justify-center"
                  >
                    <Loader2 className="w-10 h-10 text-glowLight animate-spin mb-4" />
                    <p className="text-muted text-sm">Running inference pipeline...</p>
                    <p className="text-muted/50 text-xs mt-1">Vision Encoder → BiLSTM Decoder → BiomedBERT NER</p>
                  </motion.div>
                )}

                {result && (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex-1 flex flex-col gap-4 overflow-y-auto"
                  >
                    {/* Status Bar */}
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {result.needs_review ? (
                          <AlertTriangle className="w-4 h-4 text-amber-400" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        )}
                        <span className={`text-xs font-medium ${result.needs_review ? 'text-amber-400' : 'text-emerald-400'}`}>
                          {result.needs_review ? 'Needs Review' : 'High Confidence'}
                        </span>
                      </div>
                      <span className="text-xs text-muted">{result.model_version}</span>
                    </div>

                    {/* Transcription Text */}
                    <div className="bg-white/5 rounded-xl p-4 border border-white/5">
                      <div className="text-xs text-muted uppercase tracking-wider mb-2">Raw Transcription</div>
                      <p className="text-white text-sm font-mono leading-relaxed">{result.transcription}</p>
                    </div>

                    {/* Entities */}
                    <div>
                      <div className="text-xs text-muted uppercase tracking-wider mb-3">Extracted Entities</div>
                      <div className="space-y-2">
                        {result.entities.map((entity: Entity, i: number) => (
                          <motion.div
                            key={i}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: i * 0.05 }}
                            className="flex items-center justify-between bg-white/[0.03] border border-white/5 rounded-xl px-4 py-3"
                          >
                            <div className="flex items-center gap-3">
                              <span className="text-lg">{entityIcon(entity.type)}</span>
                              <div>
                                <div className="text-white text-sm font-medium">{entity.value}</div>
                                <div className="text-muted text-xs capitalize">{entity.type}</div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full ${confidenceBg(entity.confidence)}`}
                                  style={{ width: `${entity.confidence * 100}%` }}
                                />
                              </div>
                              <span className={`text-xs font-mono ${confidenceColor(entity.confidence)}`}>
                                {(entity.confidence * 100).toFixed(1)}%
                              </span>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

        </div>
      </div>
    </div>
  );
}
