import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { listTranscriptions, type TranscriptionListResponse } from '../lib/api';
import { Loader2, FileText, Clock, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function DashboardPage() {
  const [data, setData] = useState<TranscriptionListResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listTranscriptions()
      .then(setData)
      .catch((err: any) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="min-h-screen pt-28 pb-16 px-4 relative z-10">
      <div className="max-w-5xl mx-auto">

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-semibold gradient-text mb-3">Dashboard</h1>
          <p className="text-muted">Your transcription history and clinical insights.</p>
        </motion.div>

        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-8 h-8 text-glowLight animate-spin mb-4" />
            <p className="text-muted text-sm">Loading transcriptions...</p>
          </div>
        )}

        {error && (
          <div className="glass-panel rounded-2xl p-6 text-center">
            <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto mb-3" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {data && (
          <>
            {/* Stats Bar */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
                className="glass-panel rounded-2xl p-5"
              >
                <div className="text-xs text-muted uppercase tracking-wider mb-1">Total Transcriptions</div>
                <div className="text-3xl font-semibold text-white">{data.total}</div>
              </motion.div>
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
                className="glass-panel rounded-2xl p-5"
              >
                <div className="text-xs text-muted uppercase tracking-wider mb-1">Avg. Entities / Scan</div>
                <div className="text-3xl font-semibold text-white">
                  {data.total > 0
                    ? (data.results.reduce((sum, r) => sum + r.response.entities.length, 0) / data.results.length).toFixed(1)
                    : '—'}
                </div>
              </motion.div>
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
                className="glass-panel rounded-2xl p-5"
              >
                <div className="text-xs text-muted uppercase tracking-wider mb-1">Needs Review</div>
                <div className="text-3xl font-semibold text-white">
                  {data.results.filter(r => r.response.needs_review).length}
                </div>
              </motion.div>
            </div>

            {/* Results List */}
            {data.results.length === 0 ? (
              <div className="glass-panel rounded-2xl p-12 text-center">
                <FileText className="w-10 h-10 text-muted mx-auto mb-4" />
                <p className="text-white font-medium mb-1">No transcriptions yet</p>
                <p className="text-muted text-sm">Upload a prescription to get started.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {data.results.map((item, i) => (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 + i * 0.05 }}
                    className="glass-panel rounded-2xl p-5 flex flex-col sm:flex-row sm:items-center gap-4"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        {item.response.needs_review ? (
                          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
                        ) : (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        )}
                        <p className="text-white text-sm font-medium truncate">
                          {item.response.transcription}
                        </p>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-muted mt-1">
                        <span className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {new Date(item.created_at).toLocaleString()}
                        </span>
                        <span>{item.response.entities.length} entities</span>
                        <span className="text-muted/50">{item.response.model_version}</span>
                      </div>
                    </div>
                    <div className="flex gap-2 flex-shrink-0">
                      {item.response.entities.slice(0, 3).map((ent, j) => (
                        <span
                          key={j}
                          className="text-xs bg-white/5 border border-white/10 px-2 py-1 rounded-lg text-glowLight"
                        >
                          {ent.value}
                        </span>
                      ))}
                      {item.response.entities.length > 3 && (
                        <span className="text-xs text-muted py-1">
                          +{item.response.entities.length - 3}
                        </span>
                      )}
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
