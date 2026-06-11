import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { BackgroundShader } from './components/BackgroundShader';
import { HeroSection } from './components/HeroSection';
import { BentoDashboard } from './components/BentoDashboard';
import { AuthProvider, useAuth } from './context/AuthContext';
import LoginPage from './pages/LoginPage';
import TranscribePage from './pages/TranscribePage';
import DashboardPage from './pages/DashboardPage';
import { Shield, LogOut } from 'lucide-react';

// ── Protected Route ──────────────────────────────────────────────────────────

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-white/20 border-t-glowLight rounded-full animate-spin" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

// ── Landing Page ─────────────────────────────────────────────────────────────

function LandingPage() {
  return (
    <>
      <HeroSection />
      <BentoDashboard />
    </>
  );
}

// ── Navigation ───────────────────────────────────────────────────────────────

function Navbar() {
  const { isAuthenticated, user, logout } = useAuth();
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;
  const linkClass = (path: string) =>
    `text-sm transition-colors ${isActive(path) ? 'text-white font-medium' : 'text-muted hover:text-white'}`;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 pointer-events-none">
      <Link to="/" className="flex items-center gap-2 pointer-events-auto cursor-pointer">
        <div className="w-6 h-6 bg-white rounded-full flex items-center justify-center overflow-hidden">
          <div className="w-3 h-3 bg-background rounded-full translate-x-[2px]"></div>
        </div>
        <span className="text-sm font-medium text-white hidden sm:inline">MedScript AI</span>
      </Link>

      <div className="hidden md:flex items-center gap-6 bg-surface/50 backdrop-blur-md border border-white/5 px-6 py-2 rounded-full pointer-events-auto">
        <Link to="/" className={linkClass('/')}>Home</Link>

        {isAuthenticated ? (
          <>
            <Link to="/app" className={linkClass('/app')}>Transcribe</Link>
            <Link to="/dashboard" className={linkClass('/dashboard')}>Dashboard</Link>
            <div className="w-px h-4 bg-white/10"></div>
            <span className="text-xs text-muted">{user?.username}</span>
            <button
              onClick={logout}
              className="text-muted hover:text-white transition-colors flex items-center gap-1"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </>
        ) : (
          <>
            <Link to="/login" className={linkClass('/login')}>Features</Link>
            <Link to="/login" className={linkClass('/login')}>Pricing</Link>
            <div className="w-px h-4 bg-white/10"></div>
            <Link to="/login" className="text-sm text-muted hover:text-white transition-colors flex items-center gap-2">
              Protection ↗
            </Link>
            <div className="w-6 h-6 rounded-full bg-white flex items-center justify-center">
              <Shield className="w-3 h-3 text-black" />
            </div>
          </>
        )}
      </div>

      <div className="flex items-center gap-2 pointer-events-auto cursor-pointer">
        {isAuthenticated ? (
          <Link to="/app" className="text-sm text-white font-medium bg-white/5 border border-white/10 px-4 py-1.5 rounded-full hover:bg-white/10 transition-colors">
            Open App
          </Link>
        ) : (
          <Link to="/login" className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-full border border-white/20 flex items-center justify-center overflow-hidden">
              <div className="w-3 h-3 border border-white/40 rounded-full mt-2"></div>
            </div>
            <span className="text-sm text-white font-medium">Sign In</span>
          </Link>
        )}
      </div>
    </nav>
  );
}

// ── App Root ─────────────────────────────────────────────────────────────────

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <div className="relative min-h-screen bg-background selection:bg-glowLight selection:text-black">
          <BackgroundShader />
          <Navbar />

          <main className="relative z-10 w-full overflow-x-hidden">
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route
                path="/app"
                element={
                  <ProtectedRoute>
                    <TranscribePage />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/dashboard"
                element={
                  <ProtectedRoute>
                    <DashboardPage />
                  </ProtectedRoute>
                }
              />
            </Routes>
          </main>
        </div>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
