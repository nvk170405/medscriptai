/**
 * Typed API client for the MedScript AI FastAPI backend.
 * Handles JWT injection, automatic token refresh on 401, and typed responses.
 */

const API_BASE = '/api/v1';

// ── Token helpers ────────────────────────────────────────────────────────────

export function getStoredToken(): string | null {
  return localStorage.getItem('medscript_access_token');
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem('medscript_refresh_token');
}

export function storeTokens(access: string, refresh: string) {
  localStorage.setItem('medscript_access_token', access);
  localStorage.setItem('medscript_refresh_token', refresh);
}

export function clearTokens() {
  localStorage.removeItem('medscript_access_token');
  localStorage.removeItem('medscript_refresh_token');
}

// ── Core fetch wrapper ───────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  retry = true,
): Promise<T> {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData)) {
    headers['Content-Type'] = headers['Content-Type'] || 'application/json';
  }

  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });

  // If 401, try refreshing the token once
  if (res.status === 401 && retry) {
    const refreshed = await tryRefreshToken();
    if (refreshed) {
      return apiFetch<T>(path, options, false);
    }
    // Refresh failed — clear everything
    clearTokens();
    window.location.href = '/login';
    throw new Error('Session expired');
  }

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, errorBody.detail || 'Request failed');
  }

  return res.json() as Promise<T>;
}

async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getStoredRefreshToken();
  if (!refreshToken) return false;

  try {
    const res = await fetch(`${API_BASE}/auth/token/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!res.ok) return false;

    const data = await res.json();
    storeTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

// ── Error class ──────────────────────────────────────────────────────────────

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

// ── Auth endpoints ───────────────────────────────────────────────────────────

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserProfile {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: string;
  organization: string | null;
  is_active: boolean;
  created_at: string | null;
}

export interface RegisterPayload {
  username: string;
  email: string;
  password: string;
  full_name: string;
  organization?: string;
}

export async function registerUser(payload: RegisterPayload): Promise<TokenResponse> {
  return apiFetch<TokenResponse>('/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function loginUser(username: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>('/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  return apiFetch<UserProfile>('/auth/me');
}

// ── Transcription endpoints ──────────────────────────────────────────────────

export interface Entity {
  type: string;
  value: string;
  confidence: number;
}

export interface TranscriptionResponse {
  transcription: string;
  entities: Entity[];
  word_confidences: number[];
  model_version: string;
  timestamp: string;
  needs_review: boolean;
}

export interface TranscriptionListResponse {
  results: Array<{
    id: string;
    user_id: string;
    response: TranscriptionResponse;
    created_at: string;
  }>;
  total: number;
  limit: number;
  offset: number;
}

export async function transcribeImage(file: File): Promise<TranscriptionResponse> {
  const formData = new FormData();
  formData.append('file', file);

  return apiFetch<TranscriptionResponse>('/transcribe', {
    method: 'POST',
    body: formData,
  });
}

export async function listTranscriptions(limit = 20, offset = 0): Promise<TranscriptionListResponse> {
  return apiFetch<TranscriptionListResponse>(`/transcriptions?limit=${limit}&offset=${offset}`);
}

// ── Health endpoint ──────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  model_loaded: boolean;
  uptime_seconds: number;
}

export async function checkHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health');
}

// ── Feedback endpoints ───────────────────────────────────────────────────────

export interface FeedbackPayload {
  transcription_id: string;
  original_text: string;
  corrected_text: string;
  corrected_entities?: Array<{ type: string; value: string; confidence: number }>;
  notes?: string;
}

export interface FeedbackItem {
  id: string;
  transcription_id: string;
  user_id: string;
  original_text: string;
  corrected_text: string;
  corrected_entities: Array<{ type: string; value: string; confidence: number }>;
  notes: string;
  created_at: string;
}

export interface FeedbackResponse {
  feedback_id: string;
  message: string;
}

export interface PendingFeedbackResponse {
  pending: FeedbackItem[];
  total: number;
}

export async function submitFeedback(payload: FeedbackPayload): Promise<FeedbackResponse> {
  return apiFetch<FeedbackResponse>('/feedback', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function listPendingFeedback(): Promise<PendingFeedbackResponse> {
  return apiFetch<PendingFeedbackResponse>('/feedback/pending');
}

// ── Collection endpoints ─────────────────────────────────────────────────────

export interface CollectionUploadResponse {
  collection_id: string;
  message: string;
}

export interface CollectionStats {
  total_images: number;
  annotated: number;
  pending_annotation: number;
  contributors: number;
}

export async function uploadCollectionImage(file: File): Promise<CollectionUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  return apiFetch<CollectionUploadResponse>('/collect/upload', {
    method: 'POST',
    body: formData,
  });
}

export async function getCollectionStats(): Promise<CollectionStats> {
  return apiFetch<CollectionStats>('/collect/stats');
}
