import React, { useState, useRef, useEffect } from 'react';
import {
  Leaf, Activity, MapPin, TrendingUp, Globe, LogOut, ChevronDown, Menu, X
} from 'lucide-react';
import { useAuth } from '../AuthContext';

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', icon: Leaf },
  { id: 'predict', label: 'Analyze Damage', icon: MapPin },
  { id: 'district', label: 'Regional Intelligence', icon: Activity },
  { id: 'historical', label: 'Historical Trends', icon: TrendingUp },
];

export default function Navbar({ activeTab, setActiveTab, geeStatus }) {
  const { user, logout } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const dropdownRef = useRef(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleTabChange = (id) => {
    setActiveTab(id);
    setMobileMenuOpen(false);
  };

  return (
    <header className="bg-emerald-900 text-white shadow-md sticky top-0 z-50 print:hidden">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <div
            className="flex items-center space-x-3 cursor-pointer flex-shrink-0"
            onClick={() => handleTabChange('overview')}
          >
            <div className="p-2 bg-emerald-800 rounded-lg flex items-center justify-center">
              <Leaf className="w-5 h-5 text-emerald-300" />
            </div>
            <div className="hidden sm:block">
              <h1 className="text-sm lg:text-base font-bold tracking-tight text-white flex items-center gap-2">
                Wheat Heatwave Risk AI
                <span className="text-[10px] px-2 py-0.5 rounded font-semibold bg-emerald-700/80 text-emerald-200 border border-emerald-600/50 hidden md:inline-block">
                  Enterprise
                </span>
              </h1>
              <p className="text-[10px] text-emerald-300 hidden lg:block">
                Satellite Agrometeorological Intelligence · NW India
              </p>
            </div>
          </div>

          {/* Desktop Nav Links */}
          <nav className="hidden lg:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  id={`nav-${item.id}`}
                  onClick={() => handleTabChange(item.id)}
                  className={`flex items-center space-x-1.5 px-3 py-2 rounded-md text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-emerald-800 text-white shadow-inner font-semibold'
                      : 'text-emerald-100 hover:bg-emerald-800/60 hover:text-white'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Right Controls */}
          <div className="flex items-center space-x-2 sm:space-x-3">
            {/* GEE Status Pill */}
            <div className="hidden sm:flex items-center space-x-1.5 text-xs bg-emerald-950/70 border border-emerald-700/60 px-2.5 py-1 rounded-full">
              <Globe className="w-3 h-3 text-emerald-300" />
              <span className="text-emerald-200 hidden md:inline">GEE:</span>
              <span className="flex h-2 w-2 relative">
                <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${geeStatus ? 'bg-emerald-400' : 'bg-amber-400'} opacity-75`} />
                <span className={`relative inline-flex rounded-full h-2 w-2 ${geeStatus ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              </span>
              <span className={`text-[10px] font-semibold ${geeStatus ? 'text-emerald-300' : 'text-amber-300'}`}>
                {geeStatus ? 'Live' : 'Connecting'}
              </span>
            </div>

            {/* User Avatar Dropdown */}
            {user && (
              <div className="relative" ref={dropdownRef}>
                <button
                  id="user-menu-btn"
                  onClick={() => setDropdownOpen(!dropdownOpen)}
                  className="flex items-center space-x-1.5 bg-emerald-800/80 hover:bg-emerald-800 border border-emerald-700/60 rounded-full pl-1 pr-2.5 py-1 transition-all duration-150"
                >
                  <div className="w-7 h-7 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center text-white font-bold text-[11px] flex-shrink-0 shadow-inner">
                    {user.avatar || user.name?.slice(0, 2).toUpperCase()}
                  </div>
                  <span className="text-xs font-semibold text-emerald-100 hidden sm:block max-w-[80px] truncate">
                    {user.name}
                  </span>
                  <ChevronDown className={`w-3 h-3 text-emerald-300 transition-transform duration-200 ${dropdownOpen ? 'rotate-180' : ''}`} />
                </button>

                {/* Dropdown */}
                {dropdownOpen && (
                  <div className="absolute right-0 top-full mt-2 w-56 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50 animate-in fade-in slide-in-from-top-2 duration-150">
                    <div className="p-4 border-b border-slate-800">
                      <p className="text-sm font-bold text-white truncate">{user.name}</p>
                      <p className="text-xs text-slate-400 truncate mt-0.5">{user.email}</p>
                      <span className="inline-block mt-2 text-[10px] font-semibold bg-emerald-900/60 text-emerald-300 border border-emerald-700/50 px-2 py-0.5 rounded-full uppercase tracking-wide">
                        {user.role}
                      </span>
                    </div>
                    <button
                      id="logout-btn"
                      onClick={() => { logout(); setDropdownOpen(false); }}
                      className="w-full flex items-center space-x-2.5 px-4 py-3 text-sm text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
                    >
                      <LogOut className="w-4 h-4 text-slate-500" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                )}
              </div>
            )}

            {/* Mobile Hamburger */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="lg:hidden p-2 text-emerald-200 hover:text-white hover:bg-emerald-800 rounded-lg transition-colors"
              aria-label="Toggle mobile menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileMenuOpen && (
        <div className="lg:hidden bg-emerald-950 border-t border-emerald-800/60">
          <div className="max-w-7xl mx-auto px-4 py-3 space-y-1">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleTabChange(item.id)}
                  className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-emerald-800 text-white font-semibold'
                      : 'text-emerald-200 hover:bg-emerald-900 hover:text-white'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </button>
              );
            })}

            {/* Mobile GEE status */}
            <div className="flex items-center space-x-2 px-4 py-2 text-xs text-emerald-400">
              <Globe className="w-3.5 h-3.5" />
              <span>GEE: {geeStatus ? 'Connected' : 'Connecting...'}</span>
              <span className={`w-2 h-2 rounded-full ${geeStatus ? 'bg-emerald-400' : 'bg-amber-400'}`} />
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
