import React, { useState, useEffect } from 'react';
import { getStates, getDistricts, getYears, getDistrictHistorical, submitPrediction } from '../api';
import { Activity, MapPin, Calendar, Thermometer, Droplets, Sun, AlertTriangle, ArrowUpRight } from 'lucide-react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';

export default function DistrictAnalysis() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('Punjab');
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('Ludhiana');
  const [years, setYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState(2022);

  const [districtData, setDistrictData] = useState([]);
  const [currentRecord, setCurrentRecord] = useState(null);
  const [currentCoords, setCurrentCoords] = useState({ lat: 30.7894, lon: 75.827 });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getStates().then(res => setStates(res.states || []));
    getYears().then(res => setYears(res.years || []));
  }, []);

  useEffect(() => {
    if (!selectedState) return;
    getDistricts(selectedState).then(res => {
      const dists = res.districts || [];
      setDistricts(dists);
      if (dists.length > 0) {
        const found = dists.find(d => d.District === selectedDistrict) || dists[0];
        setSelectedDistrict(found.District);
        setCurrentCoords({ lat: found.lat, lon: found.lon });
      }
    });
  }, [selectedState]);

  useEffect(() => {
    if (!selectedState || !selectedDistrict) return;
    setLoading(true);
    getDistrictHistorical(selectedState, selectedDistrict)
      .then(res => {
        const data = res.data || [];
        setDistrictData(data);
        const match = data.find(d => d.year === selectedYear);
        setCurrentRecord(match || (data.length > 0 ? data[data.length - 1] : null));
      })
      .finally(() => setLoading(false));

    const d = districts.find(item => item.District === selectedDistrict);
    if (d) {
      setCurrentCoords({ lat: d.lat, lon: d.lon });
    }
  }, [selectedState, selectedDistrict, selectedYear, districts]);

  const damageBadge = (label) => {
    switch (label) {
      case 'SEVERELY DAMAGED':
        return 'bg-rose-100 text-rose-800 border-rose-300';
      case 'MODERATELY DAMAGED':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'LESS DAMAGED':
        return 'bg-amber-100 text-amber-800 border-amber-300';
      default:
        return 'bg-emerald-100 text-emerald-800 border-emerald-300';
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Activity className="w-6 h-6 text-emerald-700" />
          District Vulnerability & Regional Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Examine multi-source environmental stress metrics across district agricultural zones.
        </p>
      </div>

      {/* Selectors Bar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap gap-4 items-center justify-between">
        <div className="flex flex-wrap items-center gap-4">
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

          <div>
            <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Year</label>
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(parseInt(e.target.value))}
              className="text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 font-medium text-slate-800"
            >
              {years.map(y => <option key={y} value={y}>{y}</option>)}
            </select>
          </div>
        </div>

        <div className="text-xs text-slate-500">
          Showing data for <strong className="text-emerald-800">{selectedDistrict}, {selectedState} ({selectedYear})</strong>
        </div>
      </div>

      {/* Grid: Map + District Metrics Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Map */}
        <div className="lg:col-span-7 bg-white p-5 rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
            <MapPin className="w-4 h-4 text-emerald-600" />
            District Geographic Footprint
          </h2>
          <div className="h-[360px] rounded-xl overflow-hidden border border-slate-200">
            <MapContainer
              key={`${currentCoords.lat}_${currentCoords.lon}`}
              center={[currentCoords.lat, currentCoords.lon]}
              zoom={9}
              className="w-full h-full"
            >
              <TileLayer
                attribution='&copy; OpenStreetMap'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              />
              <Marker position={[currentCoords.lat, currentCoords.lon]}>
                <Popup>
                  <div className="text-xs">
                    <strong>{selectedDistrict} Centroid</strong>
                    <p>{currentCoords.lat.toFixed(4)}°N, {currentCoords.lon.toFixed(4)}°E</p>
                  </div>
                </Popup>
              </Marker>
              <Circle
                center={[currentCoords.lat, currentCoords.lon]}
                radius={65000}
                pathOptions={{ color: '#059669', fillColor: '#10b981', fillOpacity: 0.1 }}
              />
            </MapContainer>
          </div>
        </div>

        {/* Indicators Card */}
        <div className="lg:col-span-5 space-y-5">
          {currentRecord ? (
            <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-5">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Historical Assessment</span>
                  <h3 className="text-lg font-bold text-slate-900">{selectedDistrict} — {selectedYear}</h3>
                </div>
                <span className={`text-xs font-bold px-3 py-1 rounded-full border ${damageBadge(currentRecord.damage_label)}`}>
                  {currentRecord.damage_label}
                </span>
              </div>

              {/* Indicator Metrics */}
              <div className="grid grid-cols-2 gap-3.5 text-xs">
                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Vegetation Index (NDVI)</span>
                  <span className="text-base font-extrabold text-emerald-700 font-mono">
                    {currentRecord.sat_ndvi_W !== null ? currentRecord.sat_ndvi_W : '—'}
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Daytime LST Mean</span>
                  <span className="text-base font-extrabold text-rose-700 font-mono">
                    {currentRecord.sat_lst_day_mean !== null ? `${currentRecord.sat_lst_day_mean}°C` : '—'}
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Filling Window Peak Tmax</span>
                  <span className="text-base font-extrabold text-amber-700 font-mono">
                    {currentRecord.wx_fill_tmax_max !== null ? `${currentRecord.wx_fill_tmax_max}°C` : '—'}
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Heatwave Days (≥35°C)</span>
                  <span className="text-base font-extrabold text-slate-900 font-mono">
                    {currentRecord.wx_fill_hot35_days} days
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Root-Zone Soil Moisture</span>
                  <span className="text-base font-extrabold text-blue-700 font-mono">
                    {currentRecord.wx_fill_sm_mean !== null ? `${currentRecord.wx_fill_sm_mean} m³/m³` : '—'}
                  </span>
                </div>

                <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                  <span className="text-slate-400 text-[10px] block">Filling Rainfall Total</span>
                  <span className="text-base font-extrabold text-sky-700 font-mono">
                    {currentRecord.wx_fill_rain_sum !== null ? `${currentRecord.wx_fill_rain_sum} mm` : '—'}
                  </span>
                </div>
              </div>

              {/* Agronomic insight */}
              <div className="bg-emerald-50/60 p-4 rounded-xl border border-emerald-200 text-xs text-emerald-900 space-y-1">
                <strong>Agronomic Context:</strong>
                <p className="leading-relaxed text-[11px] text-emerald-800">
                  High heatwave days coupled with soil moisture deficit accelerate canopy senescence during the terminal grain-filling phase (March–April), truncating the photosynthetic window and reducing kernel weight.
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-white p-8 rounded-2xl border border-slate-200 text-center text-xs text-slate-500">
              No historical data record found for this district-year.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
