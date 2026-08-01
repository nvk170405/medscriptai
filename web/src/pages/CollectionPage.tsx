import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { uploadCollectionImage, getCollectionStats, type CollectionStats } from '../lib/api';
import {
  Upload, Database, CheckCircle2, AlertTriangle, Image as ImageIcon,
  BarChart3, Users, Loader2, FolderOpen
} from 'lucide-react';

export default function CollectionPage() {
  const [uploads, setUploads] = useState<Array<{ name: string; id: string; status: 'uploading' | 'done' | 'error'; error?: string }>>([]);
  const [stats, setStats] = useState<CollectionStats | null>(null);
  const [isLoadingStats, setIsLoadingStats] = useState(true);

  useEffect(() => {
    getCollectionStats()
      .then(setStats)
      .catch(() => {})
      .finally(() => setIsLoadingStats(false));
  }, []);

  const refreshStats = () => {
    getCollectionStats().then(setStats).catch(() => {});
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const tempId = crypto.randomUUID();
      setUploads(prev => [...prev, { name: file.name, id: tempId, status: 'uploading' }]);

      try {
        const res = await uploadCollectionImage(file);
        setUploads(prev =>
          prev.map(u => u.id === tempId ? { ...u, id: res.collection_id, status: 'done' as const } : u)
        );
        refreshStats();
      } catch (err: any) {
        setUploads(prev =>
          prev.map(u => u.id === tempId ? { ...u, status: 'error' as const, error: err.message } : u)
        );
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] },
    maxSize: 10 * 1024 * 1024,
    multiple: true,
  });

  return (
    <div className="min-h-screen pt-28 pb-16 px-4 relative z-10">
      <div className="max-w-5xl mx-auto">

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl font-semibold gradient-text mb-3">Data Collection</h1>
          <p className="text-muted">Contribute prescription images to expand the training dataset and improve model accuracy.</p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { icon: ImageIcon, label: 'Total Images', value: stats?.total_images ?? '—', color: 'text-glowLight' },
            { icon: CheckCircle2, label: 'Annotated', value: stats?.annotated ?? '—', color: 'text-emerald-400' },
            { icon: FolderOpen, label: 'Pending', value: stats?.pending_annotation ?? '—', color: 'text-amber-400' },
            { icon: Users, label: 'Contributors', value: stats?.contributors ?? '—', color: 'text-sky-400' },
          ].map((stat, i) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className="glass-panel rounded-2xl p-4"
            >
              <stat.icon className={`w-4 h-4 ${stat.color} mb-2`} />
              <div className="text-2xl font-semibold text-white">{isLoadingStats ? '—' : stat.value}</div>
              <div className="text-xs text-muted mt-0.5">{stat.label}</div>
            </motion.div>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">

          {/* Upload Zone */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-3"
          >
            <div className="glass-panel rounded-3xl p-6 h-full">
              <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-glowLight" />
                Upload Prescription Images
              </h3>

              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-12 cursor-pointer transition-all min-h-[250px] ${
                  isDragActive
                    ? 'border-glowLight/60 bg-glowLight/5'
                    : 'border-white/10 hover:border-white/20 hover:bg-white/[0.02]'
                }`}
              >
                <input {...getInputProps()} />
                <Upload className="w-10 h-10 text-muted mb-4" />
                <p className="text-white font-medium mb-1">Drop prescription images here</p>
                <p className="text-muted text-sm text-center">
                  Supports batch upload · JPG, PNG up to 10MB each
                </p>
              </div>
            </div>
          </motion.div>

          {/* Upload History */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-2"
          >
            <div className="glass-panel rounded-3xl p-6 h-full flex flex-col">
              <h3 className="text-white font-medium mb-4 flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-glowLight" />
                Recent Uploads
              </h3>

              <div className="flex-1 space-y-2 overflow-y-auto max-h-[300px]">
                <AnimatePresence mode="popLayout">
                  {uploads.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-muted">
                      <ImageIcon className="w-8 h-8 mb-3 opacity-30" />
                      <p className="text-sm">No uploads yet this session</p>
                    </div>
                  ) : (
                    [...uploads].reverse().map(upload => (
                      <motion.div
                        key={upload.id}
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        className="flex items-center gap-3 bg-white/[0.03] border border-white/5 rounded-xl px-3 py-2.5"
                      >
                        {upload.status === 'uploading' && <Loader2 className="w-4 h-4 text-glowLight animate-spin flex-shrink-0" />}
                        {upload.status === 'done' && <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />}
                        {upload.status === 'error' && <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0" />}
                        <div className="min-w-0 flex-1">
                          <p className="text-white text-xs truncate">{upload.name}</p>
                          {upload.error && <p className="text-red-400 text-[10px] truncate">{upload.error}</p>}
                        </div>
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            </div>
          </motion.div>

        </div>
      </div>
    </div>
  );
}
