import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './AuthContext';
import Navbar from './components/Navbar';
import Overview from './pages/Overview';
import Predict from './pages/Predict';
import DistrictAnalysis from './pages/DistrictAnalysis';
import HistoricalAnalysis from './pages/HistoricalAnalysis';
import Auth from './pages/Auth';
import { getHealth } from './api';

function AppShell() {
  const { user, loading } = useAuth();
  const [activeTab, setActiveTab] = useState('overview');
  const [geeConnected, setGeeConnected] = useState(false);

  useEffect(() => {
    getHealth()
      .then((res) => {
        if (res.services && res.services.google_earth_engine === 'connected') {
          setGeeConnected(true);
        }
      })
      .catch((err) => console.error('Health check error:', err));
  }, []);

  // Show nothing while restoring session
  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="flex flex-col items-center space-y-4">
          <div className="w-10 h-10 border-2 border-emerald-500/30 border-t-emerald-500 rounded-full animate-spin" />
          <p className="text-slate-400 text-sm">Loading...</p>
        </div>
      </div>
    );
  }

  // Auth wall — show sign-in/sign-up if not authenticated
  if (!user) {
    return <Auth />;
  }

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col text-slate-800">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        geeStatus={geeConnected}
      />

      <main className="flex-1">
        {activeTab === 'overview' && (
          <Overview onStartAnalysis={() => setActiveTab('predict')} />
        )}
        {activeTab === 'predict' && <Predict />}
        {activeTab === 'district' && <DistrictAnalysis />}
        {activeTab === 'historical' && <HistoricalAnalysis />}
      </main>

      <footer className="bg-slate-900 text-slate-400 text-xs py-6 border-t border-slate-800 mt-12 print:hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <p className="font-semibold text-slate-200">
                Wheat Heatwave Risk AI — Enterprise Platform
              </p>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Satellite Agrometeorological Intelligence · Punjab, Haryana & Western UP
              </p>
            </div>
            <div className="flex items-center space-x-4 text-[11px] text-slate-500">
              <span>MODIS · ERA5-Land · CHIRPS</span>
              <span>·</span>
              <span className="text-emerald-400 font-medium">37 Districts · 2003–2024</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
