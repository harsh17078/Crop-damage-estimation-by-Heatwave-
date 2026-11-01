import React from 'react';
import { Leaf, Activity, MapPin, TrendingUp, BarChart3, History, Globe, Presentation, Code2 } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, geeStatus, pitchMode, setPitchMode }) {
  const allNavItems = [
    { id: 'overview', label: 'Overview', pitchLabel: 'Overview', icon: Leaf, showInPitch: true },
    { id: 'predict', label: 'Analyze Damage', pitchLabel: 'Analyze Damage', icon: MapPin, showInPitch: true },
    { id: 'district', label: 'District Analysis', pitchLabel: 'Regional Intelligence', icon: Activity, showInPitch: true },
    { id: 'historical', label: 'Historical Trends', pitchLabel: 'Historical Trust', icon: TrendingUp, showInPitch: true },
    { id: 'performance', label: 'Model Performance', pitchLabel: 'Model Diagnostics', icon: BarChart3, showInPitch: false },
    { id: 'history', label: 'Prediction History', pitchLabel: 'Audit History', icon: History, showInPitch: false },
  ];

  const navItems = pitchMode ? allNavItems.filter(i => i.showInPitch) : allNavItems;

  return (
    <header className="bg-emerald-900 text-white shadow-md sticky top-0 z-50 print:hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('overview')}>
            <div className="p-2 bg-emerald-800 rounded-lg flex items-center justify-center">
              <Leaf className="w-6 h-6 text-emerald-300" />
            </div>
            <div>
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-white flex items-center gap-2">
                Wheat Heatwave Risk AI
                <span className={`text-[11px] px-2 py-0.5 rounded font-semibold hidden sm:inline-block ${
                  pitchMode ? 'bg-amber-500/20 text-amber-200 border border-amber-400/40' : 'bg-emerald-800 text-emerald-200'
                }`}>
                  {pitchMode ? 'Enterprise Pitch' : 'GEE + ExtraTrees'}
                </span>
              </h1>
              <p className="text-xs text-emerald-200 hidden md:block">
                {pitchMode
                  ? 'Instant Satellite Intelligence for Agricultural Insurance & Disaster Relief'
                  : 'Multi-Source Satellite & Climate Damage Estimation (NW India)'}
              </p>
            </div>
          </div>

          {/* Nav Links */}
          <nav className="hidden lg:flex space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-emerald-800 text-white shadow-inner font-semibold'
                      : 'text-emerald-100 hover:bg-emerald-800/60 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{pitchMode ? item.pitchLabel : item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Side Controls */}
          <div className="flex items-center space-x-3">
            {/* Pitch / Full Research Mode Switch */}
            <button
              onClick={() => setPitchMode(!pitchMode)}
              className={`flex items-center space-x-1.5 text-xs px-3 py-1.5 rounded-full border transition-all shadow-sm ${
                pitchMode
                  ? 'bg-amber-400 text-amber-950 font-bold border-amber-300 hover:bg-amber-300'
                  : 'bg-emerald-800/80 text-emerald-100 border-emerald-700 hover:bg-emerald-800'
              }`}
              title={pitchMode ? 'Switch to Full Research / Dev Mode' : 'Switch to Pitch Mode'}
            >
              {pitchMode ? (
                <>
                  <Presentation className="w-3.5 h-3.5" />
                  <span>Pitch Mode</span>
                </>
              ) : (
                <>
                  <Code2 className="w-3.5 h-3.5" />
                  <span>Research Mode</span>
                </>
              )}
            </button>

            {/* GEE Status Pill */}
            <div className="flex items-center space-x-1.5 text-xs bg-emerald-950/70 border border-emerald-700/60 px-2.5 py-1 rounded-full">
              <Globe className="w-3.5 h-3.5 text-emerald-300" />
              <span className="text-emerald-200 hidden sm:inline">Earth Engine:</span>
              <span className="flex h-2 w-2 relative">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${geeStatus ? 'bg-emerald-400' : 'bg-amber-400'} opacity-75`}></span>
                <span className={`relative inline-flex rounded-full h-2 w-2 ${geeStatus ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
              </span>
              <span className={`font-semibold ${geeStatus ? 'text-emerald-300' : 'text-amber-300'}`}>
                {geeStatus ? 'Connected' : 'Connecting'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile nav bar */}
      <div className="lg:hidden border-t border-emerald-800/80 bg-emerald-950/95 overflow-x-auto">
        <div className="flex space-x-1 p-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center space-x-1 px-2.5 py-1.5 rounded text-xs whitespace-nowrap ${
                  isActive
                    ? 'bg-emerald-800 text-white font-semibold'
                    : 'text-emerald-200 hover:bg-emerald-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{pitchMode ? item.pitchLabel : item.label}</span>
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
}
