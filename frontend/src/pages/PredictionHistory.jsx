import React, { useState, useEffect } from 'react';
import { getHistory, exportHistoryCsvUrl, getPredictionById } from '../api';
import { 
  History, Download, Filter, Search, Eye, AlertTriangle, 
  CheckCircle2, X, Database, MapPin, Calendar, Clock 
} from 'lucide-react';

export default function PredictionHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterState, setFilterState] = useState('');
  const [filterDistrict, setFilterDistrict] = useState('');
  const [filterYear, setFilterYear] = useState('');

  const [selectedRecord, setSelectedRecord] = useState(null);
  const [modalLoading, setModalLoading] = useState(false);

  const fetchHistory = () => {
    setLoading(true);
    const params = {};
    if (filterState) params.state = filterState;
    if (filterDistrict) params.district = filterDistrict;
    if (filterYear) params.year = parseInt(filterYear);

    getHistory(params)
      .then(res => setHistory(res.history || []))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchHistory();
  }, [filterState, filterDistrict, filterYear]);

  const handleOpenDetail = (id) => {
    setModalLoading(true);
    getPredictionById(id)
      .then(res => setSelectedRecord(res.prediction))
      .finally(() => setModalLoading(false));
  };

  const getDamageBadge = (label) => {
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
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
            <History className="w-6 h-6 text-emerald-700" />
            Prediction Audit Trail & History
          </h1>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Persisted history of completed inferences with full feature payloads, confidence calibration, and OOD telemetry.
          </p>
        </div>

        <div>
          <a
            href={exportHistoryCsvUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center space-x-2 bg-emerald-800 hover:bg-emerald-900 text-white px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold shadow-sm transition-colors"
          >
            <Download className="w-4 h-4" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2 text-xs text-slate-500 font-semibold uppercase tracking-wider">
          <Filter className="w-4 h-4 text-emerald-700" />
          <span>Filters:</span>
        </div>

        <input
          type="text"
          placeholder="Filter State..."
          value={filterState}
          onChange={(e) => setFilterState(e.target.value)}
          className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
        />

        <input
          type="text"
          placeholder="Filter District..."
          value={filterDistrict}
          onChange={(e) => setFilterDistrict(e.target.value)}
          className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
        />

        <input
          type="number"
          placeholder="Filter Year..."
          value={filterYear}
          onChange={(e) => setFilterYear(e.target.value)}
          className="text-xs bg-slate-50 border border-slate-300 rounded-lg px-3 py-1.5 text-slate-800 w-28 focus:outline-none focus:ring-1 focus:ring-emerald-500"
        />

        {(filterState || filterDistrict || filterYear) && (
          <button
            onClick={() => { setFilterState(''); setFilterDistrict(''); setFilterYear(''); }}
            className="text-xs text-rose-600 hover:text-rose-800 font-medium"
          >
            Reset
          </button>
        )}
      </div>

      {/* Table */}
      <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-12 text-center text-xs text-slate-400">Loading history...</div>
        ) : history.length === 0 ? (
          <div className="p-12 text-center text-xs text-slate-400">
            No prediction records found matching the criteria.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">ID</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Region & Year</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Coordinates</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Predicted Class</th>
                  <th className="px-4 py-3 text-right font-semibold text-slate-600">Confidence</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-600">OOD Status</th>
                  <th className="px-4 py-3 text-left font-semibold text-slate-600">Timestamp</th>
                  <th className="px-4 py-3 text-center font-semibold text-slate-600">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {history.map((r) => (
                  <tr key={r.id} className="hover:bg-slate-50/80">
                    <td className="px-4 py-3 font-mono text-slate-400">#{r.id}</td>
                    <td className="px-4 py-3 font-medium text-slate-900">
                      {r.district}, {r.state}
                      <span className="block text-[11px] text-slate-400">Rabi {r.year}</span>
                    </td>
                    <td className="px-4 py-3 font-mono text-slate-600">
                      {r.latitude.toFixed(4)}, {r.longitude.toFixed(4)}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-block px-2.5 py-0.5 rounded-full font-bold border text-[11px] ${getDamageBadge(r.damage_label)}`}>
                        {r.damage_label}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono font-bold text-slate-800">
                      {(r.confidence_score * 100).toFixed(1)}%
                      <span className="block text-[10px] text-slate-400">{r.confidence_level}</span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      {r.ood?.is_ood ? (
                        <span className="inline-flex items-center text-amber-700 bg-amber-50 border border-amber-200 px-2 py-0.5 rounded text-[10px] font-bold">
                          OOD
                        </span>
                      ) : (
                        <span className="inline-flex items-center text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded text-[10px] font-bold">
                          In-Dist
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-slate-400 text-[11px] whitespace-nowrap">
                      {r.created_at ? new Date(r.created_at).toLocaleString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => handleOpenDetail(r.id)}
                        className="p-1.5 hover:bg-emerald-50 text-emerald-800 rounded-lg transition-colors"
                        title="View Full Details"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedRecord && (
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full p-6 shadow-2xl border border-slate-200 max-h-[90vh] overflow-y-auto space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <span className="text-xs font-mono text-slate-400">Record #{selectedRecord.id}</span>
                <h3 className="text-lg font-bold text-slate-900">
                  {selectedRecord.district}, {selectedRecord.state} ({selectedRecord.year})
                </h3>
              </div>
              <button
                onClick={() => setSelectedRecord(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-600 hover:bg-slate-100"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Assessment summary */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 font-bold">Assessment</span>
                  <p className="text-xl font-black text-slate-900">{selectedRecord.damage_label}</p>
                </div>
                <div className="text-right">
                  <span className="text-xs font-bold text-slate-700">{selectedRecord.confidence_level} Confidence</span>
                  <p className="text-base font-black font-mono text-emerald-800">
                    {(selectedRecord.confidence_score * 100).toFixed(1)}%
                  </p>
                </div>
              </div>

              {/* Coordinates & Source */}
              <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-200 text-slate-600">
                <div>Coordinates: <strong>{selectedRecord.latitude.toFixed(4)}°N, {selectedRecord.longitude.toFixed(4)}°E</strong></div>
                <div>Source: <strong>{selectedRecord.feature_status}</strong></div>
              </div>
            </div>

            {/* Probabilities */}
            {selectedRecord.probabilities && (
              <div className="space-y-1.5">
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider">Class Probabilities</h4>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                  <div className="p-2.5 bg-emerald-50 rounded-lg border border-emerald-200">
                    <span className="block text-[10px] text-emerald-800">Not Damaged</span>
                    <strong>{(selectedRecord.probabilities.not_damaged * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="p-2.5 bg-amber-50 rounded-lg border border-amber-200">
                    <span className="block text-[10px] text-amber-800">Less Damaged</span>
                    <strong>{(selectedRecord.probabilities.less_damaged * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="p-2.5 bg-orange-50 rounded-lg border border-orange-200">
                    <span className="block text-[10px] text-orange-800">Mod Damaged</span>
                    <strong>{(selectedRecord.probabilities.moderately_damaged * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="p-2.5 bg-rose-50 rounded-lg border border-rose-200">
                    <span className="block text-[10px] text-rose-800">Sev Damaged</span>
                    <strong>{(selectedRecord.probabilities.severely_damaged * 100).toFixed(1)}%</strong>
                  </div>
                </div>
              </div>
            )}

            {/* OOD telemetry */}
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs flex justify-between items-center">
              <div>
                <span>OOD Condition: </span>
                <strong className={selectedRecord.ood?.is_ood ? 'text-amber-700' : 'text-emerald-700'}>
                  {selectedRecord.ood?.is_ood ? 'Out-of-Distribution' : 'Within Reference Training Space'}
                </strong>
              </div>
              <div className="font-mono text-slate-500">
                Score: {selectedRecord.ood?.ood_score?.toFixed(2)} | Threshold: {selectedRecord.ood?.threshold?.toFixed(2)}
              </div>
            </div>

            {/* Rule warnings */}
            {selectedRecord.rule_warnings && selectedRecord.rule_warnings.length > 0 && (
              <div className="p-3 bg-rose-50 border border-rose-200 rounded-xl text-xs space-y-1">
                <span className="font-bold text-rose-800 block">Rule Warnings Triggered:</span>
                <ul className="list-disc pl-4 text-rose-900 space-y-1">
                  {selectedRecord.rule_warnings.map((w, idx) => <li key={idx}>{w}</li>)}
                </ul>
              </div>
            )}

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setSelectedRecord(null)}
                className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-semibold text-xs px-4 py-2 rounded-lg"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
