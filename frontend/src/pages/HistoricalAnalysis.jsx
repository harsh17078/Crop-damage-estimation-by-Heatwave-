import React, { useState, useEffect } from 'react';
import { getStates, getDistricts, getDistrictHistorical } from '../api';
import { TrendingUp, BarChart3, AlertOctagon, Sun, Droplets } from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis,
  Tooltip, Legend, CartesianGrid
} from 'recharts';

export default function HistoricalAnalysis() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('Haryana');
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('Hisar');
  const [historyData, setHistoryData] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getStates().then(res => setStates(res.states || []));
  }, []);

  useEffect(() => {
    if (!selectedState) return;
    getDistricts(selectedState).then(res => {
      const dists = res.districts || [];
      setDistricts(dists);
      if (dists.length > 0) {
        const found = dists.find(d => d.District === selectedDistrict) || dists[0];
        setSelectedDistrict(found.District);
      }
    });
  }, [selectedState]);

  useEffect(() => {
    if (!selectedState || !selectedDistrict) return;
    setLoading(true);
    getDistrictHistorical(selectedState, selectedDistrict)
      .then(res => setHistoryData(res.data || []))
      .finally(() => setLoading(false));
  }, [selectedState, selectedDistrict]);

  const selectClass = "text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 font-medium text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none w-full sm:w-auto";

  const ChartCard = ({ title, subtitle, icon: Icon, children }) => (
    <div className="bg-white p-4 sm:p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2 flex-1 min-w-0">
          <Icon className="w-4 h-4 flex-shrink-0" />
          <span className="truncate">{title}</span>
        </h2>
        <span className="text-[10px] text-slate-400 whitespace-nowrap flex-shrink-0">{subtitle}</span>
      </div>
      <div className="h-52 sm:h-64">
        {children}
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-emerald-700 flex-shrink-0" />
          Multi-Decadal Satellite & Meteorological Trends
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Historical climate exposure, vegetation vigor, and damage class progression across 22 years of MODIS, ERA5-Land, and CHIRPS data (2003–2024).
        </p>
      </div>

      {/* Selectors Bar */}
      <div className="bg-white p-4 sm:p-5 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex flex-col sm:flex-row flex-wrap items-start sm:items-end gap-3 sm:gap-4 justify-between">
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 w-full sm:w-auto">
            <div className="flex-1 sm:flex-none min-w-[130px]">
              <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">State</label>
              <select value={selectedState} onChange={(e) => setSelectedState(e.target.value)} className={selectClass}>
                {states.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
            <div className="flex-1 sm:flex-none min-w-[150px]">
              <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">District</label>
              <select value={selectedDistrict} onChange={(e) => setSelectedDistrict(e.target.value)} className={selectClass}>
                {districts.map(d => <option key={d.District} value={d.District}>{d.District}</option>)}
              </select>
            </div>
          </div>
          <div className="text-[11px] text-slate-500 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg self-start sm:self-auto">
            2003–2018: Training baseline · 2019–2024: Holdout evaluation
          </div>
        </div>
      </div>

      {loading && (
        <div className="text-center py-12 text-slate-400 text-sm">Loading satellite archive...</div>
      )}

      {/* Charts Grid */}
      {!loading && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 sm:gap-6">
          <ChartCard
            title="Thermal Stress: Tmax Peak & LST Day Mean (°C)"
            subtitle="ERA5-Land & MOD11A2"
            icon={Sun}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={32} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="wx_fill_tmax_max" name="Peak Tmax (°C)" stroke="#dc2626" strokeWidth={2} dot={{ r: 2 }} />
                <Line type="monotone" dataKey="sat_lst_day_mean" name="LST Day Mean (°C)" stroke="#f97316" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Heatwave Days During Filling Window (Tmax ≥ 35°C)"
            subtitle="Days per season"
            icon={AlertOctagon}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} width={28} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Bar dataKey="wx_fill_hot35_days" name="Hot Days (≥ 35°C)" fill="#ef4444" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Canopy Greenness: Post-Peak NDVI"
            subtitle="MODIS MOD13Q1"
            icon={BarChart3}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis domain={[0.2, 0.9]} tick={{ fontSize: 10 }} width={32} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="sat_ndvi_W" name="Filling Canopy NDVI" stroke="#10b981" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Soil Moisture & Buffering Capacity (m³/m³)"
            subtitle="ERA5-Land Root-Zone"
            icon={Droplets}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10 }} width={38} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="wx_fill_sm_mean" name="Root-Zone SM (m³/m³)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 2 }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}
    </div>
  );
}
