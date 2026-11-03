import React, { useState, useEffect } from 'react';
import { getStates, getDistricts, getYears, getDistrictHistorical } from '../api';
import { Activity, MapPin } from 'lucide-react';
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
    if (d) setCurrentCoords({ lat: d.lat, lon: d.lon });
  }, [selectedState, selectedDistrict, selectedYear, districts]);

  const damageBadge = (label) => {
    switch (label) {
      case 'SEVERELY DAMAGED': return 'bg-rose-100 text-rose-800 border-rose-300';
      case 'MODERATELY DAMAGED': return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'LESS DAMAGED': return 'bg-amber-100 text-amber-800 border-amber-300';
      default: return 'bg-emerald-100 text-emerald-800 border-emerald-300';
    }
  };

  const selectClass = "text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 font-medium text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none w-full sm:w-auto";

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Activity className="w-6 h-6 text-emerald-700 flex-shrink-0" />
          District Vulnerability & Regional Analysis
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Examine multi-source environmental stress metrics across district agricultural zones.
        </p>
      </div>

      {/* Selectors Bar */}
      <div className="bg-white p-4 sm:p-5 rounded-xl border border-slate-200 shadow-sm">
        <div className="flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4 items-start sm:items-end justify-between">
          <div className="flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4 w-full sm:w-auto">
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
            <div className="flex-1 sm:flex-none min-w-[120px]">
              <label className="block text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-1">Year</label>
              <select value={selectedYear} onChange={(e) => setSelectedYear(parseInt(e.target.value))} className={selectClass}>
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
          </div>
          <div className="text-xs text-slate-500 bg-slate-50 border border-slate-200 px-3 py-2 rounded-lg whitespace-nowrap self-start sm:self-auto">
            Viewing: <strong className="text-emerald-800">{selectedDistrict}, {selectedState} ({selectedYear})</strong>
          </div>
        </div>
      </div>

      {/* Grid: Map + Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 sm:gap-8">
        {/* Map */}
        <div className="lg:col-span-7 bg-white p-4 sm:p-5 rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <h2 className="text-sm font-bold text-slate-800 flex items-center gap-2">
            <MapPin className="w-4 h-4 text-emerald-600" />
            District Geographic Footprint
          </h2>
          <div className="h-[280px] sm:h-[360px] rounded-xl overflow-hidden border border-slate-200">
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
        <div className="lg:col-span-5">
          {currentRecord ? (
            <div className="bg-white p-5 sm:p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4 h-full">
              <div className="flex items-start justify-between border-b border-slate-100 pb-3 gap-2">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Historical Assessment</span>
                  <h3 className="text-base sm:text-lg font-bold text-slate-900">{selectedDistrict} — {selectedYear}</h3>
                </div>
                <span className={`text-xs font-bold px-2.5 py-1 rounded-full border whitespace-nowrap flex-shrink-0 ${damageBadge(currentRecord.damage_label)}`}>
                  {currentRecord.damage_label}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2.5 text-xs">
                {[
                  { label: 'Vegetation Index (NDVI)', value: currentRecord.sat_ndvi_W !== null ? currentRecord.sat_ndvi_W : '—', color: 'text-emerald-700' },
                  { label: 'Daytime LST Mean', value: currentRecord.sat_lst_day_mean !== null ? `${currentRecord.sat_lst_day_mean}°C` : '—', color: 'text-rose-700' },
                  { label: 'Peak Tmax (Filling)', value: currentRecord.wx_fill_tmax_max !== null ? `${currentRecord.wx_fill_tmax_max}°C` : '—', color: 'text-amber-700' },
                  { label: 'Heatwave Days (≥35°C)', value: `${currentRecord.wx_fill_hot35_days} days`, color: 'text-slate-900' },
                  { label: 'Root-Zone Soil Moisture', value: currentRecord.wx_fill_sm_mean !== null ? `${currentRecord.wx_fill_sm_mean} m³/m³` : '—', color: 'text-blue-700' },
                  { label: 'Filling Rainfall Total', value: currentRecord.wx_fill_rain_sum !== null ? `${currentRecord.wx_fill_rain_sum} mm` : '—', color: 'text-sky-700' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                    <span className="text-slate-400 text-[10px] block leading-tight mb-0.5">{label}</span>
                    <span className={`text-base font-extrabold font-mono ${color}`}>{value}</span>
                  </div>
                ))}
              </div>

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
