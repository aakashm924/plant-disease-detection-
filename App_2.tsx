import { useState, type ChangeEvent, useEffect, useRef, useCallback } from 'react';
import {
  Camera, ArrowLeft, UploadCloud, ShieldCheck, BarChart2,
  Leaf, X, AlertTriangle, CheckCircle, Activity, FlaskConical,
  Sprout, Wind, MessageSquare, ChevronRight, RotateCcw,
  TrendingUp, Eye, Zap
} from 'lucide-react';
import ChatWidget from './ChatWidget';

// ── Types ────────────────────────────────────────────────────────────────────
interface Prediction {
  label: string;
  display_label: string;
  confidence: number;
  is_healthy: boolean;
  is_uncertain: boolean;
  confidence_gap: number;
  severity: { level: string; score: number; color: string };
  disease_info: {
    description: string;
    spread_risk: string;
    season_tip: string;
    severity_guide: Record<string, string>;
  };
  treatments: { organic: string[]; chemical: string[] };
  precautions: string[];
  top_predictions: Array<{ label: string; display_label: string; confidence: number }>;
  gradcam: string | null;
}

interface Message {
  text: string;
  sender: 'user' | 'bot';
  loading?: boolean;
}

interface Stats {
  total_scans: number;
  healthy_count: number;
  disease_count: number;
  top_diseases: Array<{ name: string; count: number }>;
}

// ── Config ───────────────────────────────────────────────────────────────────
const API_BASE = (port: number) => `http://${window.location.hostname || 'localhost'}:${port}`;
const SESSION_ID = `session_${Date.now()}`;

// ── Severity color helpers ────────────────────────────────────────────────────
const severityConfig: Record<string, { bg: string; text: string; border: string; icon: React.ReactNode }> = {
  Healthy: {
    bg: 'bg-emerald-50', text: 'text-emerald-700', border: 'border-emerald-200',
    icon: <CheckCircle size={16} className="text-emerald-500" />
  },
  Mild: {
    bg: 'bg-yellow-50', text: 'text-yellow-700', border: 'border-yellow-200',
    icon: <AlertTriangle size={16} className="text-yellow-500" />
  },
  Moderate: {
    bg: 'bg-orange-50', text: 'text-orange-700', border: 'border-orange-200',
    icon: <AlertTriangle size={16} className="text-orange-500" />
  },
  Severe: {
    bg: 'bg-red-50', text: 'text-red-700', border: 'border-red-200',
    icon: <AlertTriangle size={16} className="text-red-500" />
  },
};

// ── Confidence bar ────────────────────────────────────────────────────────────
function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? 'bg-emerald-500' : pct >= 60 ? 'bg-yellow-500' : 'bg-red-400';
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>Confidence</span>
        <span className="font-semibold">{pct}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// ── Treatment tabs ────────────────────────────────────────────────────────────
function TreatmentPanel({ treatments }: { treatments: { organic: string[]; chemical: string[] } }) {
  const [tab, setTab] = useState<'organic' | 'chemical'>('organic');
  const items = tab === 'organic' ? treatments.organic : treatments.chemical;

  if (!treatments.organic.length && !treatments.chemical.length) return null;

  return (
    <div className="rounded-xl border border-gray-100 overflow-hidden">
      <div className="flex">
        {['organic', 'chemical'].map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t as 'organic' | 'chemical')}
            className={`flex-1 py-2.5 text-sm font-semibold flex items-center justify-center gap-2 transition-colors ${
              tab === t
                ? t === 'organic'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-700 text-white'
                : 'bg-gray-50 text-gray-500 hover:bg-gray-100'
            }`}
          >
            {t === 'organic' ? <Leaf size={14} /> : <FlaskConical size={14} />}
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>
      <div className={`p-4 ${tab === 'organic' ? 'bg-emerald-50' : 'bg-slate-50'}`}>
        {items.length === 0 ? (
          <p className="text-sm text-gray-400 italic">No {tab} treatments listed.</p>
        ) : (
          <ul className="space-y-2">
            {items.map((item, i) => (
              <li key={`${item}-${i}`} className="flex items-start gap-2 text-sm text-gray-700">
                <ChevronRight size={14} className={`mt-0.5 flex-shrink-0 ${tab === 'organic' ? 'text-emerald-500' : 'text-slate-500'}`} />
                {item}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ── Spread risk badge ─────────────────────────────────────────────────────────
function SpreadBadge({ risk }: { risk: string }) {
  const color: Record<string, string> = {
    NONE: 'bg-emerald-100 text-emerald-700',
    LOW: 'bg-blue-100 text-blue-700',
    MEDIUM: 'bg-yellow-100 text-yellow-700',
    HIGH: 'bg-orange-100 text-orange-700',
    'VERY HIGH': 'bg-red-100 text-red-700',
    EXTREME: 'bg-red-200 text-red-800 font-bold animate-pulse',
    UNKNOWN: 'bg-gray-100 text-gray-600',
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs ${color[risk] || color.UNKNOWN}`}>
      <Wind size={11} /> Spread Risk: {risk}
    </span>
  );
}

// ── Stats mini-dashboard ──────────────────────────────────────────────────────
function StatsDashboard({ stats }: { stats: Stats }) {
  const healthRate = stats.total_scans > 0
    ? Math.round((stats.healthy_count / stats.total_scans) * 100)
    : 0;

  return (
    <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
      <h3 className="text-sm font-bold text-gray-700 mb-4 flex items-center gap-2">
        <TrendingUp size={16} className="text-emerald-500" /> Session Statistics
      </h3>
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          { label: 'Total Scans', value: stats.total_scans, icon: <Activity size={16} className="text-blue-500" /> },
          { label: 'Healthy', value: stats.healthy_count, icon: <CheckCircle size={16} className="text-emerald-500" /> },
          { label: 'Diseased', value: stats.disease_count, icon: <AlertTriangle size={16} className="text-red-400" /> },
        ].map(({ label, value, icon }) => (
          <div key={label} className="text-center p-3 bg-gray-50 rounded-xl">
            <div className="flex justify-center mb-1">{icon}</div>
            <div className="text-xl font-bold text-gray-800">{value}</div>
            <div className="text-xs text-gray-400">{label}</div>
          </div>
        ))}
      </div>
      {stats.total_scans > 0 && (
        <div className="mb-4">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Healthy Rate</span>
            <span>{healthRate}%</span>
          </div>
          <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-400 rounded-full transition-all duration-700"
              style={{ width: `${healthRate}%` }}
            />
          </div>
        </div>
      )}
      {stats.top_diseases.length > 0 && (
        <div>
          <p className="text-xs text-gray-400 mb-2">Top Detected Diseases</p>
          {stats.top_diseases.map(({ name, count }) => (
            <div key={name} className="flex justify-between items-center py-1.5 border-b border-gray-50 last:border-0">
              <span className="text-xs text-gray-600 truncate">{name}</span>
              <span className="text-xs font-semibold text-orange-600 bg-orange-50 px-2 py-0.5 rounded-full">{count}×</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState<'home' | 'detect' | 'result'>('home');
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [query, setQuery] = useState('');
  const [suggestions, setSuggestions] = useState<string[]>([]);

  // Camera
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Fetch stats periodically
  const fetchStats = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE(4000)}/stats`);
      if (r.ok) setStats(await r.json());
    } catch {/* ignore */}
  }, []);

  useEffect(() => {
    fetchStats();
    const id = setInterval(fetchStats, 15000);
    return () => clearInterval(id);
  }, [fetchStats]);

  // Camera cleanup
  useEffect(() => {
    return () => { cameraStream?.getTracks().forEach(t => t.stop()); };
  }, [cameraStream]);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  // ── Camera ──
  const openCamera = async () => {
    if (!window.isSecureContext) return alert('Camera requires HTTPS or localhost.');
    if (!navigator.mediaDevices?.getUserMedia) return alert('Camera not supported in this browser.');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
        .catch(() => navigator.mediaDevices.getUserMedia({ video: true }));
      setCameraStream(stream);
      setCameraOpen(true);
      setTimeout(() => { if (videoRef.current) videoRef.current.srcObject = stream; }, 100);
    } catch (e) {
      alert(`Camera error: ${e instanceof Error ? e.message : 'Unknown error'}`);
    }
  };

  const closeCamera = () => {
    cameraStream?.getTracks().forEach(t => t.stop());
    setCameraStream(null);
    setCameraOpen(false);
  };

  const captureImage = () => {
    const video = videoRef.current;
    if (!video || !cameraStream) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0);
    canvas.toBlob(blob => {
      if (!blob) return;
      const f = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
      setFile(f);
      setPreviewUrl(canvas.toDataURL('image/jpeg'));
      closeCamera();
      setScreen('detect');
    }, 'image/jpeg', 0.92);
  };

  // ── File selection ──
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setSuggestions([]);
  };

  // ── Upload & predict ──
  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setSuggestions([]);

    const form = new FormData();
    form.append('file', file);

    try {
      const r = await fetch(`${API_BASE(4000)}/predict`, { method: 'POST', body: form });
      const data = await r.json();
      if (!r.ok || data.error) throw new Error(data.error || 'Analysis failed');
      setResult(data);
      setScreen('result');
      fetchStats();

      // Auto-fetch suggestions based on detected disease
      if (!data.is_healthy) {
        const context = `${data.display_label}: ${data.disease_info?.description || ''}`;
        try {
          const sr = await fetch(`${API_BASE(3000)}/suggest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ context }),
          });
          if (sr.ok) {
            const sd = await sr.json();
            setSuggestions(sd.suggestions || []);
          }
        } catch {/* ignore */}
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // ── Chatbot ──
  const askBot = async (queryText?: string) => {
    const q = queryText || query;
    if (!q.trim()) return;
    const userMsg = q.trim();
    setMessages(prev => [...prev, { text: userMsg, sender: 'user' }, { text: '...', sender: 'bot', loading: true }]);
    setQuery('');

    try {
      const r = await fetch(`${API_BASE(3000)}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, session_id: SESSION_ID }),
      });
      const data = await r.json();
      setMessages(prev => [...prev.slice(0, -1), { text: data.answer || 'Sorry, no answer.', sender: 'bot' }]);
    } catch {
      setMessages(prev => [...prev.slice(0, -1), { text: 'Connection error. Is the chatbot server running?', sender: 'bot' }]);
    }
  };

  const reset = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setSuggestions([]);
    setScreen('detect');
  };

  // ─────────────────────────────────────────────────────────────────────────────
  // HOME SCREEN
  // ─────────────────────────────────────────────────────────────────────────────
  if (screen === 'home') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-[#0a2e1a] via-[#0f4a28] to-[#0a2e1a] relative overflow-hidden">
        {/* Decorative circles */}
        <div className="absolute top-[-80px] right-[-80px] w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl" />
        <div className="absolute bottom-40 left-[-60px] w-64 h-64 bg-emerald-400/10 rounded-full blur-3xl" />

        {/* Header */}
        <header className="relative flex items-center justify-between px-6 pt-8 pb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-500 rounded-xl flex items-center justify-center shadow-lg">
              <Sprout size={22} className="text-white" />
            </div>
            <div>
              <span className="text-white font-bold text-xl tracking-tight">Plant AI</span>
              <div className="text-emerald-400 text-xs">Disease Detection System</div>
            </div>
          </div>
          <button
            type="button"
            onClick={() => setChatOpen(true)}
            className="flex items-center gap-2 bg-emerald-500/20 border border-emerald-500/30 text-emerald-300 px-4 py-2 rounded-full text-sm hover:bg-emerald-500/30 transition-colors"
          >
            <MessageSquare size={14} /> Ask PlantDoc
          </button>
        </header>

        {/* Hero */}
        <div className="relative px-6 py-12 text-center">
          <div className="inline-flex items-center gap-2 bg-emerald-500/15 border border-emerald-500/25 rounded-full px-4 py-1.5 text-emerald-300 text-sm mb-6">
            <Zap size={13} /> Powered by EfficientNet + Claude AI
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-white leading-tight mb-4">
            Detect Plant Diseases<br />
            <span className="text-emerald-400">Instantly & Precisely</span>
          </h1>
          <p className="text-gray-400 text-lg mb-8 max-w-md mx-auto">
            Upload a leaf photo and get expert diagnosis, treatment plans, and prevention tips in seconds.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <button
              type="button"
              onClick={() => setScreen('detect')}
              className="bg-emerald-500 hover:bg-emerald-400 text-white font-semibold px-8 py-4 rounded-2xl shadow-lg shadow-emerald-900/40 transition-all hover:shadow-emerald-800/50 hover:-translate-y-0.5"
            >
              Start Detection →
            </button>
            <button
              type="button"
              onClick={openCamera}
              className="flex items-center justify-center gap-2 border border-emerald-500/40 text-emerald-300 hover:text-white hover:border-emerald-400 font-semibold px-8 py-4 rounded-2xl transition-colors"
            >
              <Camera size={18} /> Use Camera
            </button>
          </div>
        </div>

        {/* Feature cards */}
        <div className="px-6 pb-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 max-w-4xl mx-auto">
            {[
              { icon: <Eye size={20} />, title: 'Grad-CAM', desc: 'Visual AI explanation' },
              { icon: <Activity size={20} />, title: 'Severity Score', desc: 'Mild to Severe rating' },
              { icon: <Leaf size={20} />, title: 'Organic Tips', desc: 'Natural treatments first' },
              { icon: <MessageSquare size={20} />, title: 'PlantDoc AI', desc: 'Claude-powered chat' },
            ].map(({ icon, title, desc }) => (
              <div key={title} className="bg-white/5 border border-white/10 rounded-2xl p-4 text-center hover:bg-white/8 transition-colors">
                <div className="w-10 h-10 bg-emerald-500/20 rounded-xl flex items-center justify-center mx-auto mb-2 text-emerald-400">
                  {icon}
                </div>
                <div className="text-white text-sm font-semibold">{title}</div>
                <div className="text-gray-400 text-xs mt-0.5">{desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Session stats */}
        {stats && stats.total_scans > 0 && (
          <div className="px-6 pb-8 max-w-4xl mx-auto">
            <StatsDashboard stats={stats} />
          </div>
        )}

        {/* Camera modal */}
        <CameraModal
          open={cameraOpen}
          videoRef={videoRef}
          cameraStream={cameraStream}
          onCapture={captureImage}
          onClose={closeCamera}
        />

        <ChatWidget
          isOpen={chatOpen}
          onClose={() => setChatOpen(false)}
          onOpen={() => setChatOpen(true)}
          messages={messages}
          query={query}
          setQuery={setQuery}
          askBot={askBot}
        />
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // DETECT / UPLOAD SCREEN
  // ─────────────────────────────────────────────────────────────────────────────
  if (screen === 'detect') {
    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-100 px-4 py-4 flex items-center gap-3 shadow-sm">
          <button
            type="button"
            onClick={() => setScreen('home')}
            className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          >
            <ArrowLeft size={20} className="text-gray-600" />
          </button>
          <div className="flex items-center gap-2">
            <Sprout size={20} className="text-emerald-600" />
            <span className="font-bold text-gray-800">Plant AI</span>
          </div>
        </header>

        <div className="max-w-lg mx-auto px-4 py-8">
          <h1 className="text-2xl font-bold text-gray-800 mb-1">Analyze Your Plant</h1>
          <p className="text-gray-500 text-sm mb-6">Upload a clear photo of the affected leaf for best results.</p>

          {/* Upload area */}
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-6 mb-4">
            <div className="flex gap-3 mb-5">
              <button
                type="button"
                onClick={openCamera}
                className="flex-1 flex items-center justify-center gap-2 bg-emerald-600 text-white py-3 rounded-xl font-semibold hover:bg-emerald-700 transition-colors"
              >
                <Camera size={18} /> Take Photo
              </button>
              <label className="flex-1 flex items-center justify-center gap-2 border-2 border-dashed border-emerald-300 text-emerald-600 py-3 rounded-xl font-semibold cursor-pointer hover:bg-emerald-50 transition-colors">
                <UploadCloud size={18} /> Upload
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            </div>

            {previewUrl ? (
              <div className="relative">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full h-56 object-cover rounded-xl border border-gray-100"
                />
                <button
                  type="button"
                  onClick={() => { setFile(null); setPreviewUrl(null); setError(null); }}
                  className="absolute top-2 right-2 bg-white/90 rounded-full p-1.5 shadow hover:bg-white transition-colors"
                >
                  <X size={14} className="text-gray-600" />
                </button>
              </div>
            ) : (
              <label className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-gray-200 rounded-xl text-gray-400 cursor-pointer hover:border-emerald-300 hover:text-emerald-500 transition-colors">
                <UploadCloud size={32} className="mb-2" />
                <span className="text-sm">Drop image here or click to browse</span>
                <span className="text-xs mt-1">JPG · PNG · WEBP — max 5 MB</span>
                <input type="file" className="hidden" accept="image/*" onChange={handleFileChange} />
              </label>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-4 flex items-start gap-3">
              <AlertTriangle size={18} className="text-red-500 flex-shrink-0 mt-0.5" />
              <div>
                <p className="text-red-700 font-semibold text-sm">Analysis Failed</p>
                <p className="text-red-600 text-sm mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {/* Tips */}
          <div className="bg-emerald-50 rounded-xl p-4 mb-6 border border-emerald-100">
            <p className="text-emerald-700 text-sm font-semibold mb-2">📷 Tips for best accuracy</p>
            <ul className="text-emerald-600 text-xs space-y-1">
              <li>• Leaf should fill most of the frame</li>
              <li>• Use natural daylight — avoid flash glare</li>
              <li>• Capture the most affected area</li>
              <li>• Avoid blurry or dark images</li>
            </ul>
          </div>

          <button
            type="button"
            onClick={analyze}
            disabled={!file || loading}
            className={`w-full py-4 rounded-2xl font-bold text-lg transition-all ${
              !file || loading
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-emerald-600 text-white hover:bg-emerald-700 shadow-lg shadow-emerald-200 hover:-translate-y-0.5'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center gap-3">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Analyzing with AI...
              </span>
            ) : 'Detect Disease'}
          </button>
        </div>

        <CameraModal
          open={cameraOpen}
          videoRef={videoRef}
          cameraStream={cameraStream}
          onCapture={captureImage}
          onClose={closeCamera}
        />

        <ChatWidget
          isOpen={chatOpen}
          onClose={() => setChatOpen(false)}
          onOpen={() => setChatOpen(true)}
          messages={messages}
          query={query}
          setQuery={setQuery}
          askBot={askBot}
        />
      </div>
    );
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // RESULT SCREEN
  // ─────────────────────────────────────────────────────────────────────────────
  if (screen === 'result' && result) {
    const sev = severityConfig[result.severity.level] || severityConfig.Mild;

    return (
      <div className="min-h-screen bg-gray-50">
        <header className="bg-white border-b border-gray-100 px-4 py-4 flex items-center justify-between shadow-sm">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setScreen('detect')}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <ArrowLeft size={20} className="text-gray-600" />
            </button>
            <div className="flex items-center gap-2">
              <Sprout size={20} className="text-emerald-600" />
              <span className="font-bold text-gray-800">Analysis Result</span>
            </div>
          </div>
          <button
            type="button"
            onClick={reset}
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-emerald-600 transition-colors"
          >
            <RotateCcw size={14} /> New Scan
          </button>
        </header>

        <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">

          {/* Hero result card */}
          <div className={`rounded-2xl p-5 border ${sev.bg} ${sev.border}`}>
            <div className="flex items-start justify-between mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  {sev.icon}
                  <span className={`text-xs font-bold uppercase tracking-wider ${sev.text}`}>
                    {result.severity.level}
                  </span>
                </div>
                <h2 className="text-xl font-bold text-gray-800">{result.display_label}</h2>
              </div>
              <SpreadBadge risk={result.disease_info.spread_risk} />
            </div>

            <ConfidenceBar value={result.confidence} />

            {result.is_uncertain && (
              <div className="mt-3 flex items-start gap-2 bg-amber-50 rounded-lg p-3 border border-amber-200">
                <AlertTriangle size={15} className="text-amber-500 flex-shrink-0 mt-0.5" />
                <p className="text-amber-700 text-xs">Low confidence — try a clearer, closer photo of the affected leaf.</p>
              </div>
            )}
          </div>

          {/* Description */}
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
            <h3 className="text-sm font-bold text-gray-700 mb-2 flex items-center gap-2">
              <ShieldCheck size={16} className="text-emerald-500" /> About this Condition
            </h3>
            <p className="text-gray-600 text-sm leading-relaxed">{result.disease_info.description}</p>
            {result.disease_info.season_tip && (
              <div className="mt-3 bg-sky-50 rounded-lg p-3 border border-sky-100">
                <p className="text-sky-700 text-xs">🌿 <strong>Seasonal Tip:</strong> {result.disease_info.season_tip}</p>
              </div>
            )}
          </div>

          {/* Grad-CAM */}
          {result.gradcam && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                <Eye size={16} className="text-purple-500" /> AI Attention Heatmap (Grad-CAM)
              </h3>
              <div className="rounded-xl overflow-hidden border border-gray-100">
                <img
                  src={`data:image/jpeg;base64,${result.gradcam}`}
                  alt="Grad-CAM Heatmap"
                  className="w-full object-cover"
                />
              </div>
              <p className="text-xs text-gray-400 mt-2 text-center italic">
                Red/yellow areas show exactly where the AI detected disease signs
              </p>
            </div>
          )}

          {/* Treatment tabs */}
          {(result.treatments.organic.length > 0 || result.treatments.chemical.length > 0) && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                <FlaskConical size={16} className="text-blue-500" /> Treatment Options
              </h3>
              <TreatmentPanel treatments={result.treatments} />
            </div>
          )}

          {/* Precautions */}
          {result.precautions.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                <AlertTriangle size={16} className="text-amber-500" /> Precautions
              </h3>
              <ul className="space-y-2">
                {result.precautions.map((p, i) => (
                  <li key={`${p}-${i}`} className="flex items-start gap-2 text-sm text-gray-600">
                    <ChevronRight size={14} className="text-amber-400 flex-shrink-0 mt-0.5" /> {p}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Top predictions */}
          <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
            <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
              <BarChart2 size={16} className="text-blue-500" /> Top Predictions
            </h3>
            <div className="space-y-3">
              {result.top_predictions.map((pred, i) => (
                <div key={pred.label}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className={`font-medium ${i === 0 ? 'text-gray-800' : 'text-gray-500'}`}>
                      {i + 1}. {pred.display_label}
                    </span>
                    <span className={`font-semibold ${i === 0 ? 'text-emerald-600' : 'text-gray-400'}`}>
                      {(pred.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${i === 0 ? 'bg-emerald-500' : 'bg-gray-300'}`}
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* AI Suggestions */}
          {suggestions.length > 0 && (
            <div className="bg-white rounded-2xl border border-gray-100 shadow-sm p-5">
              <h3 className="text-sm font-bold text-gray-700 mb-3 flex items-center gap-2">
                <MessageSquare size={16} className="text-indigo-500" /> Ask PlantDoc AI
              </h3>
              <div className="space-y-2">
                {suggestions.map((s, i) => (
                  <button
                    key={`sug-${i}`}
                    type="button"
                    onClick={() => { setChatOpen(true); askBot(s); }}
                    className="w-full text-left bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-sm px-4 py-2.5 rounded-xl border border-indigo-100 transition-colors"
                  >
                    💬 {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Stats */}
          {stats && <StatsDashboard stats={stats} />}

          {/* CTA */}
          <div className="flex gap-3 pb-4">
            <button
              type="button"
              onClick={reset}
              className="flex-1 py-3.5 bg-emerald-600 text-white font-semibold rounded-xl hover:bg-emerald-700 transition-colors"
            >
              Analyze Another Plant
            </button>
            <button
              type="button"
              onClick={() => setChatOpen(true)}
              className="flex-1 py-3.5 border border-emerald-200 text-emerald-700 font-semibold rounded-xl hover:bg-emerald-50 transition-colors flex items-center justify-center gap-2"
            >
              <MessageSquare size={16} /> Ask PlantDoc
            </button>
          </div>
        </div>

        <ChatWidget
          isOpen={chatOpen}
          onClose={() => setChatOpen(false)}
          onOpen={() => setChatOpen(true)}
          messages={messages}
          query={query}
          setQuery={setQuery}
          askBot={askBot}
        />
      </div>
    );
  }

  return null;
}

// ── Camera modal (shared) ─────────────────────────────────────────────────────
function CameraModal({
  open, videoRef, cameraStream, onCapture, onClose
}: {
  open: boolean;
  videoRef: React.RefObject<HTMLVideoElement>;
  cameraStream: MediaStream | null;
  onCapture: () => void;
  onClose: () => void;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl p-5 w-full max-w-sm shadow-2xl">
        <div className="flex justify-between items-center mb-4">
          <h3 className="font-bold text-gray-800">Take Photo</h3>
          <button type="button" onClick={onClose} className="p-1.5 hover:bg-gray-100 rounded-full transition-colors">
            <X size={18} className="text-gray-600" />
          </button>
        </div>
        <video ref={videoRef} autoPlay playsInline muted className="w-full rounded-xl mb-4 bg-black" />
        <p className="text-xs text-gray-500 text-center mb-4">Position the leaf in the center under good lighting</p>
        <div className="flex gap-3">
          <button
            type="button"
            onClick={onCapture}
            disabled={!cameraStream}
            className="flex-1 bg-emerald-600 text-white py-3 rounded-xl font-semibold hover:bg-emerald-700 disabled:opacity-50 transition-colors"
          >
            Capture
          </button>
          <button
            type="button"
            onClick={onClose}
            className="flex-1 bg-gray-100 text-gray-700 py-3 rounded-xl font-semibold hover:bg-gray-200 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
