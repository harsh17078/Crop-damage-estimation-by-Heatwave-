import React, { useState, useEffect } from 'react';
import {
  getStates, getDistricts, getYears, submitPrediction
} from '../api';
import MapPicker from '../components/MapPicker';
import ProbabilityBar from '../components/ProbabilityBar';
import FeaturesTable from '../components/FeaturesTable';
import {
  Play, AlertCircle, CheckCircle, AlertTriangle, ShieldAlert,
  MapPin, RefreshCw, ThermometerSun,
  Printer, CheckCircle2, Zap, Lightbulb, Compass
} from 'lucide-react';

const DAMAGE_STYLE = {
  0: {
    label: 'NOT DAMAGED',
    tier: 'Healthy Crop Baseline',
    color: '#16a34a',
    bg: 'bg-emerald-50 border-emerald-300 text-emerald-900',
    badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
  },
  1: {
    label: 'LESS DAMAGED',
    tier: 'Low Damage Risk',
    color: '#eab308',
    bg: 'bg-amber-50 border-amber-300 text-amber-900',
    badge: 'bg-amber-100 text-amber-800 border-amber-300',
  },
  2: {
    label: 'MODERATELY DAMAGED',
    tier: 'Moderate Damage Risk',
    color: '#f97316',
    bg: 'bg-orange-50 border-orange-300 text-orange-900',
    badge: 'bg-orange-100 text-orange-800 border-orange-300',
  },
  3: {
    label: 'SEVERELY DAMAGED',
    tier: 'Severe Crop Loss Risk',
    color: '#dc2626',
    bg: 'bg-rose-50 border-rose-300 text-rose-900',
    badge: 'bg-rose-100 text-rose-800 border-rose-300',
  },
};

const ACTIONABLE_ADVISORIES = {
  0: {
    status: 'Healthy Canopy & Normal Thermal Range',
    action: 'Maintain standard seasonal irrigation and fertilization schedule. No emergency insurance claim triggers required.',
    priority: 'Routine Monitoring',
    priorityColor: 'text-emerald-700 bg-emerald-100',
    icon: CheckCircle2,
  },
  1: {
    status: 'Sub-Critical Heat Stress Detected',
    action: 'Schedule follow-up satellite scan in 7 days. Recommend light evening irrigation to replenish topsoil moisture and prevent canopy dehydration.',
    priority: 'Low Priority Advisory',
    priorityColor: 'text-amber-800 bg-amber-100',
    icon: Lightbulb,
  },
  2: {
    status: 'Significant Evaporative Deficit & Senescence',
    action: 'Trigger localized irrigation support. Alert local Krishi Vigyan Kendra (KVK) agronomists for canopy cooling advisories; flag for preliminary insurance inspection.',
    priority: 'Urgent Farm Advisory',
    priorityColor: 'text-orange-800 bg-orange-100',
    icon: AlertTriangle,
  },
  3: {
    status: 'Acute Heat Desiccation & Early Maturity Trigger',
    action: 'Coordinate with district insurance desks for expedited claim processing. Expedite ground-validation fast-track and dispatch state disaster relief logistics.',
    priority: 'High Priority Action',
    priorityColor: 'text-rose-800 bg-rose-100',
    icon: ShieldAlert,
  },
};

export default function Predict() {
  const [states, setStates] = useState([]);
  const [selectedState, setSelectedState] = useState('Haryana');
  const [districts, setDistricts] = useState([]);
  const [selectedDistrict, setSelectedDistrict] = useState('Bhiwani');
  const [years, setYears] = useState([]);
  const [selectedYear, setSelectedYear] = useState(2022);

  const [latitude, setLatitude] = useState(28.7328);
  const [longitude, setLongitude] = useState(75.8928);
  const [currentCentroid, setCurrentCentroid] = useState(null);

  const [loading, setLoading] = useState(false);
  const [progressStep, setProgressStep] = useState('');
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    getStates().then(data => {
      if (data.states?.length > 0) setStates(data.states);
    }).catch(console.error);

    getYears().then(data => {
      if (data.years?.length > 0) setYears(data.years);
    }).catch(console.error);
  }, []);

  useEffect(() => {
    if (!selectedState) return;
    getDistricts(selectedState).then(data => {
      if (data.districts?.length > 0) {
        setDistricts(data.districts);
        const exists = data.districts.find(d => d.District === selectedDistrict);
        const target = exists || data.districts[0];
        setSelectedDistrict(target.District);
        setLatitude(target.lat);
        setLongitude(target.lon);
        setCurrentCentroid({ lat: target.lat, lon: target.lon });
      }
    }).catch(console.error);
  }, [selectedState]);

  const handleDistrictChange = (distName) => {
    setSelectedDistrict(distName);
    const d = districts.find(item => item.District === distName);
    if (d) {
      setLatitude(d.lat);
      setLongitude(d.lon);
      setCurrentCentroid({ lat: d.lat, lon: d.lon });
    }
  };

  const handleMapCoordChange = (newLat, newLon) => {
    setLatitude(newLat);
    setLongitude(newLon);
  };

  const handlePrintReport = () => window.print();

  const handleAnalyze = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      setProgressStep('Connecting to Google Earth Engine satellite feeds...');
      await new Promise(r => setTimeout(r, 450));
      setProgressStep('Querying MODIS (LST/NDVI) & ERA5-Land climate parameters...');
      await new Promise(r => setTimeout(r, 650));
      setProgressStep('Synthesizing 89 agro-climatic features & running ExtraTrees classifier...');

      const response = await submitPrediction({
        state: selectedState,
        district: selectedDistrict,
        year: parseInt(selectedYear),
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
      });

      if (response.success) {
        setResult(response);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.error || err.message || 'Server error during extraction');
    } finally {
      setLoading(false);
      setProgressStep('');
    }
  };

  const isFutureYear = selectedYear > 2024;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6 sm:space-y-8">

      {/* Page Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
            <Compass className="w-6 h-6 text-emerald-700 flex-shrink-0" />
            Instant Crop Damage Verification
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Select location parameters or pin an inspection point. GEE extracts multi-sensor data to generate an audit-ready damage verdict in seconds.
          </p>
        </div>
        <div className="flex-shrink-0 flex items-center space-x-2 text-xs bg-emerald-50 border border-emerald-200 text-emerald-800 px-3.5 py-1.5 rounded-full font-medium shadow-sm self-start sm:self-auto">
          <Zap className="w-3.5 h-3.5 text-emerald-600" />
          <span>Turnaround: <strong>~5 Seconds</strong></span>
        </div>
      </div>

      {/* Main Grid: Form/Map (left) + Results (right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">

        {/* Left: Input Form & Map */}
        <div className="lg:col-span-7 space-y-5 print:hidden">
          <form onSubmit={handleAnalyze} className="bg-white rounded-2xl p-5 sm:p-6 border border-slate-200 shadow-sm space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <h2 className="text-sm sm:text-base font-bold text-slate-800 flex items-center gap-2">
                <MapPin className="w-4 h-4 text-emerald-600" />
                Target Geography & Season
              </h2>
              <span className="text-[11px] text-slate-400 font-medium">Step 1 of 2</span>
            </div>

            {/* Dropdowns Row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">State</label>
                <select
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                  className="w-full text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {states.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">District</label>
                <select
                  value={selectedDistrict}
                  onChange={(e) => handleDistrictChange(e.target.value)}
                  className="w-full text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {districts.map(d => <option key={d.District} value={d.District}>{d.District}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Harvest Year</label>
                <select
                  value={selectedYear}
                  onChange={(e) => setSelectedYear(parseInt(e.target.value))}
                  className="w-full text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2.5 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {years.map(y => (
                    <option key={y} value={y}>{y}{y > 2024 ? ' • Live Operational' : ''}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Future Year Notice */}
            {isFutureYear && (
              <div className="flex items-start gap-2 bg-emerald-50 border border-emerald-300 text-emerald-900 text-xs p-3 rounded-lg">
                <Zap className="w-4 h-4 text-emerald-700 flex-shrink-0 mt-0.5" />
                <span>
                  <strong>Live Operational Deployment (Rabi {selectedYear}):</strong> Year {selectedYear} processes real-time GEE multi-spectral observations with automated phenological anchoring.
                </span>
              </div>
            )}

            {/* Coordinates */}
            <div className="grid grid-cols-2 gap-3 sm:gap-4 pt-1">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Latitude (°N)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={latitude}
                  onChange={(e) => setLatitude(parseFloat(e.target.value) || 0)}
                  className="w-full text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2.5 font-mono text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Longitude (°E)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={longitude}
                  onChange={(e) => setLongitude(parseFloat(e.target.value) || 0)}
                  className="w-full text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2.5 font-mono text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Interactive Map */}
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center justify-between text-xs">
                <label className="font-semibold text-slate-700">Interactive GIS Pin Drop</label>
                <span className="text-[11px] text-slate-400">Catchment radius: 65km</span>
              </div>
              <MapPicker
                latitude={latitude}
                longitude={longitude}
                districtCentroid={currentCentroid}
                districtName={selectedDistrict}
                onLocationChange={handleMapCoordChange}
              />
            </div>

            {/* Submit */}
            <button
              type="submit"
              id="analyze-btn"
              disabled={loading}
              className="w-full mt-1 bg-emerald-800 hover:bg-emerald-900 text-white font-bold py-3.5 px-4 rounded-xl shadow-md transition-all flex items-center justify-center space-x-2 disabled:opacity-60 disabled:cursor-not-allowed text-sm"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-emerald-300" />
                  <span>Synthesizing Satellite Imagery & Running ML...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Run Instant Satellite Damage Assessment</span>
                </>
              )}
            </button>

            {loading && progressStep && (
              <div className="text-center text-xs text-emerald-800 bg-emerald-50/90 border border-emerald-200 py-2.5 rounded-lg animate-pulse font-medium">
                {progressStep}
              </div>
            )}
          </form>
        </div>

        {/* Right: Results Panel */}
        <div className="lg:col-span-5 space-y-5 print:col-span-12">
          {error && (
            <div className="bg-rose-50 border border-rose-300 text-rose-900 rounded-2xl p-5 shadow-sm space-y-2">
              <div className="flex items-center space-x-2 text-rose-700 font-bold text-sm">
                <AlertCircle className="w-5 h-5 flex-shrink-0" />
                <span>Assessment Error</span>
              </div>
              <p className="text-xs text-rose-800 leading-relaxed">{error}</p>
            </div>
          )}

          {result ? (
            <div className="space-y-4 sm:space-y-5">
              {/* Primary Damage Card */}
              {(() => {
                const style = DAMAGE_STYLE[result.prediction.damage_class_id] || DAMAGE_STYLE[0];
                const advisory = ACTIONABLE_ADVISORIES[result.prediction.damage_class_id] || ACTIONABLE_ADVISORIES[0];

                return (
                  <div className={`rounded-2xl p-5 sm:p-6 border shadow-md space-y-4 ${style.bg}`}>
                    <div className="flex items-start justify-between border-b border-slate-200/60 pb-3 gap-2">
                      <div>
                        <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                          Assessed Damage Severity Tier
                        </span>
                        <h2 className="text-xl sm:text-2xl font-black mt-0.5 tracking-tight text-slate-900">
                          {style.label}
                        </h2>
                        <p className="text-xs text-slate-600 mt-0.5 font-semibold">{style.tier}</p>
                      </div>
                      <div className="text-right flex-shrink-0">
                        <span className={`text-xs font-bold px-2 py-1 rounded-full border whitespace-nowrap ${style.badge}`}>
                          {result.prediction.confidence_level} CONF.
                        </span>
                        <p className="text-lg font-black font-mono mt-1 text-slate-800">
                          {(result.prediction.confidence_score * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {/* Location Summary */}
                    <div className="grid grid-cols-2 gap-2 text-xs bg-white/80 p-3 rounded-xl border border-slate-200">
                      <div>
                        <span className="text-slate-400 block text-[10px]">District & State</span>
                        <strong className="text-slate-800">{result.location.district}, {result.location.state}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px]">Harvest Season</span>
                        <strong className="text-slate-800">Rabi {result.location.year}</strong>
                      </div>
                      <div>
                        <span className="text-slate-400 block text-[10px]">Coordinates</span>
                        <span className="font-mono text-slate-700 text-[11px]">
                          {result.location.latitude.toFixed(4)}°N, {result.location.longitude.toFixed(4)}°E
                        </span>
                      </div>
                      {result.prediction.pred_exposure_tmax_degC !== null && (
                        <div>
                          <span className="text-slate-400 block text-[10px]">Est. Heat Exposure</span>
                          <strong className="text-rose-700 font-mono">
                            {result.prediction.pred_exposure_tmax_degC} °C
                          </strong>
                        </div>
                      )}
                    </div>

                    {/* Probability Distribution */}
                    <div>
                      <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                        Damage Probability Breakdown
                      </h3>
                      <ProbabilityBar
                        probabilities={result.probabilities}
                        predictedClassId={result.prediction.damage_class_id}
                      />
                    </div>
                  </div>
                );
              })()}

              {/* Actionable Advisory */}
              {(() => {
                const adv = ACTIONABLE_ADVISORIES[result.prediction.damage_class_id] || ACTIONABLE_ADVISORIES[0];
                const Icon = adv.icon;
                return (
                  <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200 shadow-sm space-y-3">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <div className="flex items-center space-x-2">
                        <Icon className="w-5 h-5 text-emerald-700" />
                        <h3 className="font-bold text-sm text-slate-900">Actionable Operational Advisory</h3>
                      </div>
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${adv.priorityColor}`}>
                        {adv.priority}
                      </span>
                    </div>
                    <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-100 space-y-1.5 text-xs">
                      <p className="font-semibold text-slate-800">{adv.status}</p>
                      <p className="text-slate-600 leading-relaxed">{adv.action}</p>
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-1 pt-0.5 text-[11px] text-slate-400">
                      <span>Suitable for: Crop Insurance Desks & State Agronomists</span>
                      <span>Target: Rabi Grain Filling</span>
                    </div>
                  </div>
                );
              })()}

              {/* OOD / Microclimate Flag */}
              <div className={`p-4 rounded-xl border shadow-sm ${
                result.ood.is_ood
                  ? 'bg-amber-50/90 border-amber-300 text-amber-900'
                  : 'bg-emerald-50/60 border-emerald-200 text-emerald-900'
              }`}>
                <div className="flex items-start space-x-2.5">
                  {result.ood.is_ood ? (
                    <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
                  ) : (
                    <CheckCircle className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" />
                  )}
                  <div className="space-y-1 text-xs">
                    <p className="font-bold">
                      {result.ood.is_ood ? 'Microclimate Anomaly Flagged' : 'Baseline Climate Profile: Standard'}
                    </p>
                    <p className="leading-relaxed">
                      {result.ood.is_ood
                        ? 'This location exhibits an atypical combination of temperature, canopy desiccation, and rainfall. The assessment is valid, but flagged for priority field audit to verify local microclimate dynamics.'
                        : 'Satellite and meteorological parameters conform to established historical baseline patterns for this agro-ecological zone.'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Rule Warnings */}
              {result.rule_warnings?.length > 0 && (
                <div className="bg-rose-50 border border-rose-300 rounded-xl p-4 shadow-sm space-y-2">
                  <div className="flex items-center space-x-2 text-rose-800 font-bold text-xs uppercase tracking-wider">
                    <ShieldAlert className="w-4 h-4 text-rose-600" />
                    <span>Physical Consistency Flag</span>
                  </div>
                  <ul className="space-y-1.5 text-xs text-rose-900">
                    {result.rule_warnings.map((warn, i) => (
                      <li key={i} className="flex items-start gap-1.5">
                        <span className="text-rose-600 font-bold">•</span>
                        <span>{warn}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  type="button"
                  onClick={handlePrintReport}
                  id="print-report-btn"
                  className="w-full flex items-center justify-center space-x-2 bg-slate-900 hover:bg-slate-800 text-white font-bold py-3 px-4 rounded-xl shadow-md transition-all text-xs sm:text-sm"
                >
                  <Printer className="w-4 h-4 text-emerald-400" />
                  <span>Generate Executive PDF Report</span>
                </button>
                <FeaturesTable featureExtraction={result.feature_extraction} />
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl p-8 sm:p-10 border border-slate-200/80 shadow-sm text-center space-y-4">
              <div className="w-14 h-14 bg-emerald-50 rounded-2xl mx-auto flex items-center justify-center text-emerald-700">
                <Compass className="w-7 h-7" />
              </div>
              <h3 className="font-bold text-slate-800 text-base">Ready for Instant Assessment</h3>
              <p className="text-xs text-slate-500 max-w-xs mx-auto leading-relaxed">
                Select any district in Punjab, Haryana, or Western UP, choose a harvest year, and click the button to retrieve satellite intelligence in seconds.
              </p>
              <div className="pt-1">
                <span className="text-[11px] text-emerald-700 font-medium bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-200">
                  ⚡ 5-Second Multi-Spectral GEE Processing
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
