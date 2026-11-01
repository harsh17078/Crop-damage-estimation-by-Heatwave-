import React, { useState, useEffect } from 'react';
import { getStates, getDistricts, getDistrictHistorical } from '../api';
import { TrendingUp, BarChart3, AlertOctagon, Sun, Droplets, Info } from 'lucide-react';
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

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-emerald-700" />
          Multi-Decadal Satellite & Meteorological Trends (2003–2024)
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Historical climate exposure, vegetation vigor, and damage class progression across 22 years of MODIS, ERA5-Land, and CHIRPS data.
        </p>
      </div>

      {/* Selectors Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap gap-4 items-center justify-between">
        <div className="flex items-center gap-4">
          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">State</label>
            <select
              value={selectedState}
              onChange={(e) => setSelectedState(e.target.value)}
              className="text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 font-medium text-slate-800"
            >
              {states.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">District</label>
            <select
              value={selectedDistrict}
              onChange={(e) => setSelectedDistrict(e.target.value)}
              className="text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 font-medium text-slate-800"
            >
              {districts.map(d => <option key={d.District} value={d.District}>{d.District}</option>)}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs bg-slate-100 px-3 py-1.5 rounded-lg text-slate-600">
          <Info className="w-4 h-4 text-slate-400" />
          <span>2003–2018: Training baseline | 2019–2024: Holdout evaluation</span>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Chart 1: Thermal Stress (Tmax & LST) */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Sun className="w-4 h-4 text-amber-600" />
              Thermal Stress: Tmax Peak & LST Day Mean (°C)
            </h2>
            <span className="text-[11px] text-slate-400">ERA5-Land & MOD11A2</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="wx_fill_tmax_max" name="Peak Tmax (°C)" stroke="#dc2626" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="sat_lst_day_mean" name="LST Day Mean (°C)" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400">
            Note the prominent 2022 thermal spike during March–April terminal grain filling across North-West India.
          </p>
        </div>

        {/* Chart 2: Heatwave Days (>= 35 C) */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <AlertOctagon className="w-4 h-4 text-rose-600" />
              Heatwave Days During Filling Window (Tmax ≥ 35°C)
            </h2>
            <span className="text-[11px] text-slate-400">Days per season</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="wx_fill_hot35_days" name="Hot Days (≥ 35°C)" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400">
            Cumulative high-heat duration is a principal driver of physiological desiccation and forced early maturity.
          </p>
        </div>

        {/* Chart 3: Vegetation Health (NDVI) */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-emerald-600" />
              Canopy Greenness: Post-Peak NDVI (sat_ndvi_W)
            </h2>
            <span className="text-[11px] text-slate-400">MODIS MOD13Q1</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis domain={[0.2, 0.9]} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="sat_ndvi_W" name="Filling Canopy NDVI" stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400">
            Rapid post-peak decline in NDVI indicates premature canopy senescence caused by heat/water stress.
          </p>
        </div>

        {/* Chart 4: Soil Moisture Dynamics */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Droplets className="w-4 h-4 text-blue-600" />
              Soil Moisture & Buffering Capacity (m³/m³)
            </h2>
            <span className="text-[11px] text-slate-400">ERA5-Land Root-Zone</span>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis domain={['auto', 'auto']} tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="wx_fill_sm_mean" name="Root-Zone Soil Moisture (m³/m³)" stroke="#3b82f6" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-slate-400">
            Higher soil moisture provides evaporative cooling, buffering crops from extreme air temperatures.
          </p>
        </div>
      </div>
    </div>
  );
}
