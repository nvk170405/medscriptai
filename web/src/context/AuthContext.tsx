import React, { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import {
  type UserProfile,
  type TokenResponse,
  type RegisterPayload,
  registerUser,
  loginUser,
  fetchCurrentUser,
  storeTokens,
  clearTokens,
  getStoredToken,
} from '../lib/api';

interface AuthState {
  user: UserProfile | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthState | null>(null);

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used inside <AuthProvider>');
  return ctx;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // On mount, try to load user from existing token
  useEffect(() => {
    const token = getStoredToken();
    if (!token) {
      setIsLoading(false);
      return;
    }

    fetchCurrentUser()
      .then(setUser)
      .catch(() => {
        clearTokens();
      })
      .finally(() => setIsLoading(false));
  }, []);

  const handleTokenResponse = useCallback(async (res: TokenResponse) => {
    storeTokens(res.access_token, res.refresh_token);
    const profile = await fetchCurrentUser();
    setUser(profile);
    setError(null);
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    setError(null);
    setIsLoading(true);
    try {
      const res = await loginUser(username, password);
      await handleTokenResponse(res);
    } catch (err: any) {
      setError(err.message || 'Login failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [handleTokenResponse]);

  const register = useCallback(async (payload: RegisterPayload) => {
    setError(null);
    setIsLoading(true);
    try {
      const res = await registerUser(payload);
      await handleTokenResponse(res);
    } catch (err: any) {
      setError(err.message || 'Registration failed');
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [handleTokenResponse]);

  const logout = useCallback(() => {
    clearTokens();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        error,
        login,
        register,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}
