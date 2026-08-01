import React, { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  listTranscriptions, submitFeedback, listPendingFeedback,
  type TranscriptionListResponse, type FeedbackItem
} from '../lib/api';
import {
  Loader2, MessageSquarePlus, CheckCircle2, AlertTriangle,
  ClipboardEdit, Clock, ChevronRight, Send, History
} from 'lucide-react';

export default function FeedbackPage() {
  const [transcriptions, setTranscriptions] = useState<TranscriptionListResponse | null>(null);
  const [feedbackHistory, setFeedbackHistory] = useState<FeedbackItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Review panel state
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [correctedText, setCorrectedText] = useState('');
  const [notes, setNotes] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);

  useEffect(() => {
    Promise.all([
      listTranscriptions(50),
      listPendingFeedback(),
    ])
      .then(([t, f]) => {
        setTranscriptions(t);
        setFeedbackHistory(f.pending);
      })
      .catch((err: any) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, []);

  const selectedItem = transcriptions?.results.find(r => r.id === selectedId);

  const handleSelect = useCallback((id: string) => {
    const item = transcriptions?.results.find(r => r.id === id);
    if (!item) return;
    setSelectedId(id);
    setCorrectedText(item.response.transcription);
    setNotes('');
    setSubmitSuccess(false);
  }, [transcriptions]);

  const handleSubmit = async () => {
    if (!selectedItem) return;
    setIsSubmitting(true);
    setSubmitSuccess(false);

    try {
      await submitFeedback({
        transcription_id: selectedItem.id,
        original_text: selectedItem.response.transcription,
        corrected_text: correctedText,
        corrected_entities: selectedItem.response.entities,
        notes,
      });
      setSubmitSuccess(true);
      // Refresh feedback history
      const f = await listPendingFeedback();
      setFeedbackHistory(f.pending);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen pt-28 pb-16 px-4 relative z-10">
      <div className="max-w-6xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-semibold gradient-text mb-3">Review & Feedback</h1>
          <p className="text-muted">Correct AI transcriptions to improve model accuracy through human-in-the-loop learning.</p>
        </motion.div>

        {isLoading && (
          <div className="flex flex-col items-center py-20">
            <Loader2 className="w-8 h-8 text-glowLight animate-spin mb-4" />
            <p className="text-muted text-sm">Loading transcriptions...</p>
          </div>
        )}

        {error && (
          <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3 mb-6">
            <AlertTriangle className="w-4 h-4 flex-shrink-0" />
            {error}
          </div>
        )}

        {!isLoading && transcriptions && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

            {/* Left — Transcription List */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              className="lg:col-span-4"
            >
              <div className="glass-panel rounded-3xl p-5 h-full flex flex-col">
                <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                  <ClipboardEdit className="w-5 h-5 text-glowLight" />
                  Transcriptions
                  <span className="ml-auto text-xs text-muted">{transcriptions.total} total</span>
                </h3>

                <div className="space-y-2 overflow-y-auto max-h-[500px] flex-1">
                  {transcriptions.results.length === 0 ? (
                    <div className="text-center py-12 text-muted text-sm">
                      No transcriptions to review yet
                    </div>
                  ) : (
                    transcriptions.results.map((item, i) => (
                      <motion.button
                        key={item.id}
                        initial={{ opacity: 0, y: 5 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.03 }}
                        onClick={() => handleSelect(item.id)}
                        className={`w-full text-left flex items-center gap-3 px-3 py-3 rounded-xl transition-all ${
                          selectedId === item.id
                            ? 'bg-glowLight/10 border border-glowLight/20'
                            : 'bg-white/[0.02] border border-white/5 hover:bg-white/[0.05]'
                        }`}
                      >
                        <div className="min-w-0 flex-1">
                          <p className="text-white text-sm truncate">{item.response.transcription}</p>
                          <div className="flex items-center gap-2 mt-1">
                            {item.response.needs_review ? (
                              <span className="text-[10px] text-amber-400 flex items-center gap-0.5">
                                <AlertTriangle className="w-2.5 h-2.5" /> Needs Review
                              </span>
                            ) : (
                              <span className="text-[10px] text-emerald-400 flex items-center gap-0.5">
                                <CheckCircle2 className="w-2.5 h-2.5" /> High Conf
                              </span>
                            )}
                            <span className="text-[10px] text-muted">{item.response.entities.length} entities</span>
                          </div>
                        </div>
                        <ChevronRight className="w-4 h-4 text-muted flex-shrink-0" />
                      </motion.button>
                    ))
                  )}
                </div>
              </div>
            </motion.div>

            {/* Right — Review Panel */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-8"
            >
              <div className="glass-panel rounded-3xl p-6 min-h-[500px] flex flex-col">
                <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                  <MessageSquarePlus className="w-5 h-5 text-glowLight" />
                  Correction Editor
                </h3>

                <AnimatePresence mode="wait">
                  {!selectedItem ? (
                    <motion.div
                      key="empty"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex-1 flex flex-col items-center justify-center text-muted"
                    >
                      <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center mb-4">
                        <ClipboardEdit className="w-8 h-8" />
                      </div>
                      <p className="text-sm">Select a transcription to review</p>
                    </motion.div>
                  ) : (
                    <motion.div
                      key={selectedId}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="flex-1 flex flex-col gap-5"
                    >
                      {/* Original Text */}
                      <div>
                        <div className="text-xs text-muted uppercase tracking-wider mb-2">Original Transcription</div>
                        <div className="bg-white/5 rounded-xl p-4 border border-white/5">
                          <p className="text-white/70 text-sm font-mono">{selectedItem.response.transcription}</p>
                        </div>
                      </div>

                      {/* Entities Preview */}
                      <div>
                        <div className="text-xs text-muted uppercase tracking-wider mb-2">Extracted Entities</div>
                        <div className="flex flex-wrap gap-2">
                          {selectedItem.response.entities.map((ent, i) => (
                            <span
                              key={i}
                              className="text-xs bg-white/5 border border-white/10 px-2.5 py-1 rounded-lg text-glowLight"
                            >
                              {ent.value} <span className="text-muted">({ent.type})</span>
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Corrected Text */}
                      <div>
                        <div className="text-xs text-muted uppercase tracking-wider mb-2">Corrected Transcription</div>
                        <textarea
                          value={correctedText}
                          onChange={e => setCorrectedText(e.target.value)}
                          rows={4}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-sm font-mono resize-none focus:outline-none focus:border-glowLight/30 transition-colors"
                          placeholder="Edit the transcription here..."
                        />
                      </div>

                      {/* Notes */}
                      <div>
                        <div className="text-xs text-muted uppercase tracking-wider mb-2">Reviewer Notes (optional)</div>
                        <input
                          type="text"
                          value={notes}
                          onChange={e => setNotes(e.target.value)}
                          className="w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-glowLight/30 transition-colors"
                          placeholder="e.g. Medication name was misread due to poor handwriting..."
                        />
                      </div>

                      {/* Submit */}
                      <div className="flex items-center gap-3 mt-auto">
                        <button
                          onClick={handleSubmit}
                          disabled={isSubmitting}
                          className="flex items-center gap-2 bg-white text-black font-medium px-6 py-3 rounded-xl hover:bg-gray-200 transition-colors disabled:opacity-50"
                        >
                          {isSubmitting ? (
                            <Loader2 className="w-4 h-4 animate-spin" />
                          ) : (
                            <Send className="w-4 h-4" />
                          )}
                          Submit Correction
                        </button>
                        {submitSuccess && (
                          <motion.span
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            className="text-emerald-400 text-sm flex items-center gap-1"
                          >
                            <CheckCircle2 className="w-4 h-4" /> Submitted
                          </motion.span>
                        )}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Feedback History */}
              {feedbackHistory.length > 0 && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="glass-panel rounded-3xl p-5 mt-6"
                >
                  <h3 className="text-white font-medium mb-3 flex items-center gap-2">
                    <History className="w-5 h-5 text-glowLight" />
                    Correction History
                    <span className="ml-auto text-xs text-muted">{feedbackHistory.length} corrections</span>
                  </h3>
                  <div className="space-y-2 max-h-[200px] overflow-y-auto">
                    {feedbackHistory.map((fb, i) => (
                      <div key={fb.id} className="flex items-start gap-3 bg-white/[0.02] border border-white/5 rounded-xl px-3 py-2.5">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
                        <div className="min-w-0 flex-1">
                          <p className="text-white text-xs truncate">{fb.corrected_text}</p>
                          <div className="flex items-center gap-3 text-[10px] text-muted mt-1">
                            <span className="flex items-center gap-0.5">
                              <Clock className="w-2.5 h-2.5" />
                              {new Date(fb.created_at).toLocaleString()}
                            </span>
                            {fb.notes && <span className="truncate">Note: {fb.notes}</span>}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </motion.div>

          </div>
        )}
      </div>
    </div>
  );
}
