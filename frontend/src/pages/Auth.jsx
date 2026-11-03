import React, { useState } from 'react';
import { useAuth } from '../AuthContext';
import { Eye, EyeOff, Leaf, Satellite, Shield, Zap, ArrowRight, Loader2 } from 'lucide-react';

const DEMO_EMAIL = 'demo@wheatai.com';
const DEMO_PASSWORD = 'demo1234';

function InputField({ id, label, type = 'text', value, onChange, placeholder, error, rightElement }) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="block text-xs font-semibold text-slate-300 uppercase tracking-wider">
        {label}
      </label>
      <div className="relative">
        <input
          id={id}
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          autoComplete={type === 'password' ? 'current-password' : 'email'}
          className={`w-full bg-white/5 border rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 
            focus:outline-none focus:ring-2 transition-all duration-200
            ${error
              ? 'border-rose-500/70 focus:ring-rose-500/40 focus:border-rose-500'
              : 'border-white/10 focus:ring-emerald-500/40 focus:border-emerald-500/60'
            }`}
        />
        {rightElement && (
          <div className="absolute right-3 top-1/2 -translate-y-1/2">{rightElement}</div>
        )}
      </div>
      {error && <p className="text-xs text-rose-400 mt-1">{error}</p>}
    </div>
  );
}

export default function Auth() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup'
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState({});
  const [serverError, setServerError] = useState('');

  const switchMode = (newMode) => {
    setMode(newMode);
    setErrors({});
    setServerError('');
    setName('');
    setEmail('');
    setPassword('');
  };

  const fillDemo = () => {
    setEmail(DEMO_EMAIL);
    setPassword(DEMO_PASSWORD);
    setErrors({});
    setServerError('');
    if (mode !== 'signin') setMode('signin');
  };

  const validate = () => {
    const e = {};
    if (mode === 'signup' && !name.trim()) e.name = 'Full name is required.';
    if (!email.trim()) e.email = 'Email address is required.';
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) e.email = 'Enter a valid email address.';
    if (!password) e.password = 'Password is required.';
    else if (mode === 'signup' && password.length < 6) e.password = 'Password must be at least 6 characters.';
    return e;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setServerError('');
    const validationErrors = validate();
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      return;
    }
    setErrors({});
    setLoading(true);
    try {
      if (mode === 'signin') {
        await login(email, password);
      } else {
        await register(name, email, password);
      }
    } catch (err) {
      setServerError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    { icon: Satellite, text: 'Live Google Earth Engine integration' },
    { icon: Zap, text: 'Sub-5-second satellite damage assessments' },
    { icon: Shield, text: 'Tamper-proof audit trail & PDF export' },
  ];

  return (
    <div className="min-h-screen flex bg-slate-950 overflow-hidden">
      {/* ─── Left Branding Panel ─── */}
      <div className="hidden lg:flex lg:w-[52%] xl:w-[55%] flex-col justify-between relative overflow-hidden">
        {/* Animated background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-950 via-emerald-900 to-teal-950" />
        <div className="absolute inset-0">
          <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-emerald-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '6s' }} />
          <div className="absolute bottom-[-20%] right-[-10%] w-[70%] h-[70%] bg-teal-500/8 rounded-full blur-3xl animate-pulse" style={{ animationDuration: '9s', animationDelay: '2s' }} />
          <div className="absolute top-[40%] right-[20%] w-[40%] h-[40%] bg-emerald-400/6 rounded-full blur-2xl animate-pulse" style={{ animationDuration: '7s', animationDelay: '1s' }} />
        </div>

        {/* Grid overlay */}
        <div className="absolute inset-0 opacity-[0.04]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.8) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.8) 1px, transparent 1px)`,
            backgroundSize: '40px 40px'
          }}
        />

        {/* Content */}
        <div className="relative z-10 flex flex-col h-full p-10 xl:p-14">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-emerald-500/20 border border-emerald-400/30 rounded-xl flex items-center justify-center backdrop-blur-sm">
              <Leaf className="w-5 h-5 text-emerald-300" />
            </div>
            <div>
              <p className="text-white font-bold text-sm tracking-tight">Wheat Heatwave Risk AI</p>
              <p className="text-emerald-400 text-[10px] font-medium tracking-widest uppercase">Enterprise Platform</p>
            </div>
          </div>

          {/* Hero Text */}
          <div className="flex-1 flex flex-col justify-center space-y-8 max-w-lg">
            <div className="space-y-4">
              <div className="inline-flex items-center space-x-2 bg-emerald-500/10 border border-emerald-400/20 text-emerald-300 text-xs font-semibold px-3 py-1.5 rounded-full">
                <Satellite className="w-3.5 h-3.5" />
                <span>Satellite Agrometeorological Intelligence</span>
              </div>
              <h1 className="text-3xl xl:text-4xl font-black text-white leading-tight">
                Instant Satellite-Powered<br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 to-teal-300">
                  Wheat Crop Damage
                </span><br />
                Assessment
              </h1>
              <p className="text-sm text-emerald-100/70 leading-relaxed">
                Connecting Google Earth Engine with ensemble Machine Learning to detect 
                terminal heat stress and crop loss across <strong className="text-emerald-300">Punjab, Haryana & Western UP</strong> in seconds.
              </p>
            </div>

            {/* Feature bullets */}
            <ul className="space-y-3.5">
              {features.map(({ icon: Icon, text }) => (
                <li key={text} className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-emerald-500/15 border border-emerald-400/20 rounded-lg flex items-center justify-center flex-shrink-0">
                    <Icon className="w-4 h-4 text-emerald-400" />
                  </div>
                  <span className="text-sm text-emerald-100/80 font-medium">{text}</span>
                </li>
              ))}
            </ul>

            {/* Stats strip */}
            <div className="grid grid-cols-3 gap-4 border-t border-emerald-700/40 pt-6">
              {[
                { value: '37', label: 'Districts' },
                { value: '22yr', label: 'Archive' },
                { value: '~5s', label: 'Assessment' },
              ].map(({ value, label }) => (
                <div key={label} className="text-center">
                  <p className="text-2xl font-black text-white">{value}</p>
                  <p className="text-[11px] text-emerald-400/70 font-medium uppercase tracking-wider mt-0.5">{label}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Bottom badge */}
          <div className="text-[11px] text-emerald-500/60 flex items-center space-x-2">
            <Shield className="w-3.5 h-3.5" />
            <span>Enterprise Grade • MODIS • ERA5-Land • CHIRPS • ExtraTrees ML</span>
          </div>
        </div>
      </div>

      {/* ─── Right Auth Panel ─── */}
      <div className="flex-1 flex flex-col items-center justify-center p-6 sm:p-10 lg:p-12 relative">
        {/* Mobile logo */}
        <div className="lg:hidden flex items-center space-x-2 mb-8">
          <div className="w-8 h-8 bg-emerald-500/20 border border-emerald-500/30 rounded-lg flex items-center justify-center">
            <Leaf className="w-4 h-4 text-emerald-400" />
          </div>
          <span className="text-white font-bold text-sm">Wheat Heatwave Risk AI</span>
        </div>

        <div className="w-full max-w-md">
          {/* Tab switcher */}
          <div className="flex bg-white/5 border border-white/10 rounded-2xl p-1 mb-8">
            <button
              type="button"
              onClick={() => switchMode('signin')}
              className={`flex-1 text-sm font-semibold py-2.5 rounded-xl transition-all duration-200 ${
                mode === 'signin'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => switchMode('signup')}
              className={`flex-1 text-sm font-semibold py-2.5 rounded-xl transition-all duration-200 ${
                mode === 'signup'
                  ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-900/50'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Create Account
            </button>
          </div>

          {/* Heading */}
          <div className="mb-7">
            <h2 className="text-2xl font-black text-white">
              {mode === 'signin' ? 'Welcome back' : 'Get started'}
            </h2>
            <p className="text-sm text-slate-400 mt-1">
              {mode === 'signin'
                ? 'Sign in to access your satellite intelligence dashboard.'
                : 'Create your account to start crop damage assessments.'}
            </p>
          </div>

          {/* Server error */}
          {serverError && (
            <div className="mb-5 p-3.5 bg-rose-500/10 border border-rose-500/30 rounded-xl text-sm text-rose-400 font-medium">
              {serverError}
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5" noValidate>
            {mode === 'signup' && (
              <div
                className="overflow-hidden transition-all duration-300"
                style={{ maxHeight: mode === 'signup' ? '200px' : '0' }}
              >
                <InputField
                  id="auth-name"
                  label="Full Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="e.g. Arjun Sharma"
                  error={errors.name}
                />
              </div>
            )}

            <InputField
              id="auth-email"
              label="Email Address"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@organisation.com"
              error={errors.email}
            />

            <InputField
              id="auth-password"
              label="Password"
              type={showPassword ? 'text' : 'password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === 'signup' ? 'Min. 6 characters' : '••••••••'}
              error={errors.password}
              rightElement={
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="text-slate-500 hover:text-slate-300 transition-colors p-0.5"
                  tabIndex={-1}
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              }
            />

            {/* Submit */}
            <button
              id="auth-submit-btn"
              type="submit"
              disabled={loading}
              className="w-full bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 disabled:cursor-not-allowed text-white font-bold py-3.5 px-6 rounded-xl shadow-lg shadow-emerald-900/40 transition-all duration-200 flex items-center justify-center space-x-2 text-sm hover:shadow-emerald-800/50 hover:scale-[1.01] active:scale-[0.99]"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>{mode === 'signin' ? 'Signing in...' : 'Creating account...'}</span>
                </>
              ) : (
                <>
                  <span>{mode === 'signin' ? 'Sign In to Dashboard' : 'Create Account'}</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Divider */}
          <div className="my-6 flex items-center gap-3">
            <div className="flex-1 h-px bg-white/10" />
            <span className="text-xs text-slate-600 font-medium">or</span>
            <div className="flex-1 h-px bg-white/10" />
          </div>

          {/* Demo shortcut */}
          <button
            id="auth-demo-btn"
            type="button"
            onClick={fillDemo}
            className="w-full bg-white/5 hover:bg-white/10 border border-white/10 hover:border-emerald-500/30 text-slate-300 hover:text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 flex items-center justify-center space-x-2 text-sm"
          >
            <Zap className="w-4 h-4 text-emerald-400" />
            <span>Use Demo Credentials</span>
          </button>

          {mode === 'signin' && (
            <p className="text-center text-xs text-slate-600 mt-4">
              Demo: <span className="font-mono text-slate-500">demo@wheatai.com</span> / <span className="font-mono text-slate-500">demo1234</span>
            </p>
          )}

          {/* Footer */}
          <p className="text-center text-xs text-slate-600 mt-8">
            {mode === 'signin' ? (
              <>Don't have an account?{' '}
                <button type="button" onClick={() => switchMode('signup')} className="text-emerald-400 hover:text-emerald-300 font-semibold transition-colors">
                  Create one
                </button>
              </>
            ) : (
              <>Already have an account?{' '}
                <button type="button" onClick={() => switchMode('signin')} className="text-emerald-400 hover:text-emerald-300 font-semibold transition-colors">
                  Sign in
                </button>
              </>
            )}
          </p>
        </div>
      </div>
    </div>
  );
}
