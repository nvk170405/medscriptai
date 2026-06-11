import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';
import { ArrowRight, AlertCircle, User, Mail, Lock, Building } from 'lucide-react';

export default function LoginPage() {
  const { login, register, error, isLoading } = useAuth();
  const navigate = useNavigate();
  const [isRegister, setIsRegister] = useState(false);

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [organization, setOrganization] = useState('');
  const [localError, setLocalError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLocalError('');

    try {
      if (isRegister) {
        await register({ username, email, password, full_name: fullName, organization: organization || undefined });
      } else {
        await login(username, password);
      }
      navigate('/app');
    } catch (err: any) {
      setLocalError(err.message || 'Something went wrong');
    }
  };

  const displayError = localError || error;

  return (
    <div className="min-h-screen flex items-center justify-center px-4 relative z-10">
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md"
      >
        <div className="glass-panel rounded-3xl p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="w-12 h-12 rounded-2xl bg-glowLight/10 border border-glowLight/20 flex items-center justify-center mx-auto mb-4">
              <span className="text-glowLight text-xl font-bold">M</span>
            </div>
            <h1 className="text-2xl font-semibold text-white mb-1">
              {isRegister ? 'Create Account' : 'Welcome Back'}
            </h1>
            <p className="text-muted text-sm">
              {isRegister ? 'Start digitizing prescriptions today' : 'Sign in to your MedScript AI account'}
            </p>
          </div>

          {/* Error */}
          {displayError && (
            <div className="flex items-center gap-2 bg-red-500/10 border border-red-500/20 text-red-400 text-sm rounded-xl px-4 py-3 mb-6">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              {displayError}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs text-muted uppercase tracking-wider font-medium">Username</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                <input
                  type="text" value={username} onChange={e => setUsername(e.target.value)}
                  required minLength={3}
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-muted text-sm focus:outline-none focus:border-glowLight/40 transition-colors"
                  placeholder="e.g. dr_navketan"
                />
              </div>
            </div>

            {isRegister && (
              <>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted uppercase tracking-wider font-medium">Email</label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                    <input
                      type="email" value={email} onChange={e => setEmail(e.target.value)}
                      required
                      className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-muted text-sm focus:outline-none focus:border-glowLight/40 transition-colors"
                      placeholder="you@hospital.com"
                    />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted uppercase tracking-wider font-medium">Full Name</label>
                  <div className="relative">
                    <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                    <input
                      type="text" value={fullName} onChange={e => setFullName(e.target.value)}
                      required
                      className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-muted text-sm focus:outline-none focus:border-glowLight/40 transition-colors"
                      placeholder="Dr. Navketan Singh"
                    />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <label className="text-xs text-muted uppercase tracking-wider font-medium">Organization (Optional)</label>
                  <div className="relative">
                    <Building className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                    <input
                      type="text" value={organization} onChange={e => setOrganization(e.target.value)}
                      className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-muted text-sm focus:outline-none focus:border-glowLight/40 transition-colors"
                      placeholder="AIIMS Delhi"
                    />
                  </div>
                </div>
              </>
            )}

            <div className="space-y-1.5">
              <label className="text-xs text-muted uppercase tracking-wider font-medium">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
                <input
                  type="password" value={password} onChange={e => setPassword(e.target.value)}
                  required minLength={8}
                  className="w-full bg-white/5 border border-white/10 rounded-xl pl-10 pr-4 py-3 text-white placeholder-muted text-sm focus:outline-none focus:border-glowLight/40 transition-colors"
                  placeholder="••••••••"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full mt-2 flex items-center justify-center gap-2 bg-white text-black font-medium py-3 rounded-xl hover:bg-gray-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
              ) : (
                <>
                  {isRegister ? 'Create Account' : 'Sign In'}
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Toggle */}
          <div className="mt-6 text-center">
            <button
              onClick={() => { setIsRegister(!isRegister); setLocalError(''); }}
              className="text-sm text-muted hover:text-white transition-colors"
            >
              {isRegister ? 'Already have an account? Sign In' : "Don't have an account? Register"}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
