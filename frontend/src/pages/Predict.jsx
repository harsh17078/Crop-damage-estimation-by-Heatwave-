import React, { useState, useEffect } from 'react';
import { 
  getStates, getDistricts, getYears, submitPrediction 
} from '../api';
import MapPicker from '../components/MapPicker';
import ProbabilityBar from '../components/ProbabilityBar';
import FeaturesTable from '../components/FeaturesTable';
import { 
  Play, AlertCircle, CheckCircle, AlertTriangle, ShieldAlert, 
  MapPin, Calendar, Compass, RefreshCw, ThermometerSun, Info
} from 'lucide-react';

const DAMAGE_STYLE = {
  0: {
    label: 'NOT DAMAGED',
    scientific: 'No Significant Heatwave Damage',
    color: '#16a34a',
    bg: 'bg-emerald-50 border-emerald-300 text-emerald-900',
    badge: 'bg-emerald-100 text-emerald-800 border-emerald-300',
    iconColor: 'text-emerald-600',
  },
  1: {
    label: 'LESS DAMAGED',
    scientific: 'Mild Heatwave Damage',
    color: '#eab308',
    bg: 'bg-amber-50 border-amber-300 text-amber-900',
    badge: 'bg-amber-100 text-amber-800 border-amber-300',
    iconColor: 'text-amber-600',
  },
  2: {
    label: 'MODERATELY DAMAGED',
    scientific: 'Moderate Heatwave Damage',
    color: '#f97316',
    bg: 'bg-orange-50 border-orange-300 text-orange-900',
    badge: 'bg-orange-100 text-orange-800 border-orange-300',
    iconColor: 'text-orange-600',
  },
  3: {
    label: 'SEVERELY DAMAGED',
    scientific: 'Severe Heatwave Damage',
    color: '#dc2626',
    bg: 'bg-rose-50 border-rose-300 text-rose-900',
    badge: 'bg-rose-100 text-rose-800 border-rose-300',
    iconColor: 'text-rose-600',
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

  // Load States & Years on Mount
  useEffect(() => {
    getStates().then(data => {
      if (data.states && data.states.length > 0) {
        setStates(data.states);
      }
    }).catch(err => console.error(err));

    getYears().then(data => {
      if (data.years && data.years.length > 0) {
        setYears(data.years);
      }
    }).catch(err => console.error(err));
  }, []);

  // Update Districts when State Changes
  useEffect(() => {
    if (!selectedState) return;
    getDistricts(selectedState).then(data => {
      if (data.districts && data.districts.length > 0) {
        setDistricts(data.districts);
        // Default to first district or retain if available
        const exists = data.districts.find(d => d.District === selectedDistrict);
        const target = exists || data.districts[0];
        setSelectedDistrict(target.District);
        setLatitude(target.lat);
        setLongitude(target.lon);
        setCurrentCentroid({ lat: target.lat, lon: target.lon });
      }
    }).catch(err => console.error(err));
  }, [selectedState]);

  // Update coordinates when District dropdown changes
  const handleDistrictChange = (distName) => {
    setSelectedDistrict(distName);
    const d = districts.find(item => item.District === distName);
    if (d) {
      setLatitude(d.lat);
      setLongitude(d.lon);
      setCurrentCentroid({ lat: d.lat, lon: d.lon });
    }
  };

  // Map clicks/drag callback
  const handleMapCoordChange = (newLat, newLon) => {
    setLatitude(newLat);
    setLongitude(newLon);
  };

  // Submit Analysis
  const handleAnalyze = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setLoading(true);

    try {
      setProgressStep('Validating coordinates & district catchment...');
      await new Promise(r => setTimeout(r, 400));

      setProgressStep('Querying Google Earth Engine (MODIS, ERA5-Land, CHIRPS)...');
      await new Promise(r => setTimeout(r, 600));

      setProgressStep('Executing ExtraTreesClassifier & OOD Mahalanobis check...');
      const response = await submitPrediction({
        state: selectedState,
        district: selectedDistrict,
        year: parseInt(selectedYear),
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude)
      });

      if (response.success) {
        setResult(response);
      } else {
        setError(response.error || 'Prediction failed');
      }
    } catch (err) {
      console.error(err);
      const errMsg = err.response?.data?.error || err.message || 'Server error during extraction';
      setError(errMsg);
    } finally {
      setLoading(false);
      setProgressStep('');
    }
  };

  const isHoldoutYear = selectedYear >= 2019 && selectedYear <= 2024;
  const isFutureYear = selectedYear > 2024;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Page Title */}
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <Compass className="w-6 h-6 text-emerald-700" />
          Location-Based Wheat Crop Damage Assessment
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Select an administrative region or drag the interactive map marker. Features are extracted directly via GEE and classified by the trained ExtraTrees model.
        </p>
      </div>

      {/* Main Grid: Form/Map + Results */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input Form & Map (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <form onSubmit={handleAnalyze} className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-5">
            <h2 className="text-base font-bold text-slate-800 border-b border-slate-100 pb-3 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-emerald-600" />
              Location & Season Parameters
            </h2>

            {/* Dropdowns Row */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              {/* State */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">State</label>
                <select
                  value={selectedState}
                  onChange={(e) => setSelectedState(e.target.value)}
                  className="w-full text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {states.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>

              {/* District */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">District</label>
                <select
                  value={selectedDistrict}
                  onChange={(e) => handleDistrictChange(e.target.value)}
                  className="w-full text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {districts.map(d => <option key={d.District} value={d.District}>{d.District}</option>)}
                </select>
              </div>

              {/* Year */}
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Harvest Year</label>
                <select
                  value={selectedYear}
                  onChange={(e) => setSelectedYear(parseInt(e.target.value))}
                  className="w-full text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                >
                  {years.map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </div>
            </div>

            {/* Year Warnings */}
            {isFutureYear && (
              <div className="flex items-start gap-2 bg-amber-50 border border-amber-300 text-amber-800 text-xs p-3 rounded-lg">
                <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <span>
                  <strong>Unseen Year Notice:</strong> Year {selectedYear} is outside the validated training/holdout period (2003–2024). Extrapolated satellite observations may trigger OOD warnings.
                </span>
              </div>
            )}
            {!isHoldoutYear && !isFutureYear && (
              <div className="flex items-start gap-2 bg-blue-50 border border-blue-200 text-blue-800 text-xs p-2.5 rounded-lg">
                <Info className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" />
                <span>Model training period (2003–2018). Results represent historical training space.</span>
              </div>
            )}

            {/* Manual Lat/Lon Coordinates */}
            <div className="grid grid-cols-2 gap-4 pt-1">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Latitude (°N)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={latitude}
                  onChange={(e) => setLatitude(parseFloat(e.target.value) || 0)}
                  className="w-full text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 font-mono text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Longitude (°E)</label>
                <input
                  type="number"
                  step="0.0001"
                  value={longitude}
                  onChange={(e) => setLongitude(parseFloat(e.target.value) || 0)}
                  className="w-full text-xs sm:text-sm bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 font-mono text-slate-800 focus:ring-2 focus:ring-emerald-500 focus:outline-none"
                />
              </div>
            </div>

            {/* Interactive Leaflet Map */}
            <div className="space-y-1.5 pt-2">
              <label className="block text-xs font-semibold text-slate-700">
                Interactive Map (Click or Drag Marker to Refine)
              </label>
              <MapPicker
                latitude={latitude}
                longitude={longitude}
                districtCentroid={currentCentroid}
                districtName={selectedDistrict}
                onLocationChange={handleMapCoordChange}
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full mt-2 bg-emerald-800 hover:bg-emerald-900 text-white font-bold py-3.5 px-4 rounded-xl shadow-md transition-all flex items-center justify-center space-x-2 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-emerald-300" />
                  <span className="text-sm">Processing Satellite Intelligence...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span className="text-sm">Analyze Crop Damage</span>
                </>
              )}
            </button>

            {/* Progress status */}
            {loading && progressStep && (
              <div className="text-center text-xs text-emerald-800 bg-emerald-50/80 border border-emerald-200 py-2 rounded-lg animate-pulse font-medium">
                {progressStep}
              </div>
            )}
          </form>
        </div>

        {/* Right Column: Prediction Results (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
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
            <div className="space-y-5">
              {/* Primary Damage Classification Card */}
              {(() => {
                const style = DAMAGE_STYLE[result.prediction.damage_class_id] || DAMAGE_STYLE[0];
                return (
                  <div className={`rounded-2xl p-6 border shadow-md space-y-5 ${style.bg}`}>
                    <div className="flex items-center justify-between border-b border-slate-200/60 pb-3">
                      <div>
                        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-500">
                          Crop Damage Assessment
                        </span>
                        <h2 className="text-xl sm:text-2xl font-black mt-0.5 tracking-tight text-slate-900">
                          {style.label}
                        </h2>
                        <p className="text-xs text-slate-600 mt-0.5 font-medium">{style.scientific}</p>
                      </div>
                      <div className="text-right">
                        <span className={`text-xs font-bold px-2.5 py-1 rounded-full border ${style.badge}`}>
                          {result.prediction.confidence_level} CONFIDENCE
                        </span>
                        <p className="text-lg font-black font-mono mt-1 text-slate-800">
                          {(result.prediction.confidence_score * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {/* Location Summary Pill */}
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
                        <span className="font-mono text-slate-700">
                          {result.location.latitude.toFixed(4)}°N, {result.location.longitude.toFixed(4)}°E
                        </span>
                      </div>
                      {result.prediction.pred_exposure_tmax_degC !== null && (
                        <div>
                          <span className="text-slate-400 block text-[10px]">Pred Heat Exposure (Tmax)</span>
                          <strong className="text-rose-700 font-mono">
                            {result.prediction.pred_exposure_tmax_degC} °C
                          </strong>
                        </div>
                      )}
                    </div>

                    {/* Class Probabilities Distribution */}
                    <div>
                      <h3 className="text-xs font-bold text-slate-700 mb-2 uppercase tracking-wider">
                        Predicted Class Probabilities
                      </h3>
                      <ProbabilityBar
                        probabilities={result.probabilities}
                        predictedClassId={result.prediction.damage_class_id}
                      />
                    </div>
                  </div>
                );
              })()}

              {/* OOD Detection Card */}
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
                      {result.ood.is_ood ? 'Out-of-Distribution (OOD) Detected' : 'In-Distribution Sample'}
                    </p>
                    <p className="leading-relaxed">
                      {result.ood.is_ood
                        ? "⚠️ This location's satellite/weather feature profile differs from the model's training distribution (Mahalanobis distance exceeds threshold). The prediction is available but may have elevated uncertainty."
                        : "Input features are within the model's reference training distribution."}
                    </p>
                    <div className="flex items-center space-x-3 text-[11px] text-slate-600 pt-1 font-mono">
                      <span>OOD Score: <strong>{result.ood.ood_score.toFixed(2)}</strong></span>
                      <span>Threshold: <strong>{result.ood.threshold.toFixed(2)}</strong></span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Rule-Based Warnings Card (if triggered) */}
              {result.rule_warnings && result.rule_warnings.length > 0 && (
                <div className="bg-rose-50 border border-rose-300 rounded-xl p-4 shadow-sm space-y-2">
                  <div className="flex items-center space-x-2 text-rose-800 font-bold text-xs uppercase tracking-wider">
                    <ShieldAlert className="w-4 h-4 text-rose-600" />
                    <span>Rule-Based Verification Flag</span>
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

              {/* Collapsible Extracted Features */}
              <FeaturesTable featureExtraction={result.feature_extraction} />
            </div>
          ) : (
            <div className="bg-white rounded-2xl p-8 border border-slate-200/80 shadow-sm text-center space-y-4">
              <div className="w-14 h-14 bg-emerald-50 rounded-2xl mx-auto flex items-center justify-center text-emerald-700">
                <Compass className="w-7 h-7" />
              </div>
              <h3 className="font-bold text-slate-800 text-base">Ready for Location Analysis</h3>
              <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                Choose a location in Haryana, Punjab, or Western UP, select a harvest year, and click 
                <strong> Analyze Crop Damage</strong> to trigger live Google Earth Engine data extraction.
              </p>
              <div className="pt-2">
                <span className="text-[11px] text-emerald-700 font-medium bg-emerald-50 px-3 py-1.5 rounded-full border border-emerald-200">
                  ⚡ Real-time MODIS & ERA5-Land Integration
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
