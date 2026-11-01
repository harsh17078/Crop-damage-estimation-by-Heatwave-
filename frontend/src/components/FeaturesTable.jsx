import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Satellite, Database, Clock } from 'lucide-react';

export default function FeaturesTable({ featureExtraction }) {
  const [isOpen, setIsOpen] = useState(false);

  if (!featureExtraction) return null;

  const summary = featureExtraction.features_summary || {};
  const featureKeys = Object.keys(summary);

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-3.5 bg-slate-50 hover:bg-slate-100 flex items-center justify-between transition-colors text-left"
      >
        <div className="flex items-center space-x-2.5">
          <Satellite className="w-4 h-4 text-emerald-700" />
          <span className="font-semibold text-sm text-slate-800">
            View Extracted Satellite & Weather Features
          </span>
          <span className="text-xs bg-emerald-100 text-emerald-800 font-medium px-2 py-0.5 rounded-full">
            {featureKeys.length} Indicators
          </span>
        </div>
        <div className="flex items-center space-x-2 text-xs text-slate-500">
          <span className="hidden sm:inline">{isOpen ? 'Collapse' : 'Expand'}</span>
          {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      {isOpen && (
        <div className="p-5 border-t border-slate-200">
          {/* Metadata badges */}
          <div className="flex flex-wrap gap-2 mb-4 text-xs">
            <div className="flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1 rounded-md">
              <Database className="w-3.5 h-3.5 text-slate-500" />
              <span>Source: <strong className="text-emerald-800">{featureExtraction.source}</strong></span>
            </div>
            {featureExtraction.extraction_time_ms && (
              <div className="flex items-center space-x-1.5 bg-slate-100 text-slate-700 px-3 py-1 rounded-md">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                <span>Extraction Time: {featureExtraction.extraction_time_ms} ms</span>
              </div>
            )}
          </div>

          {/* Table */}
          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-4 py-2.5 text-left font-semibold text-slate-700">Feature Indicator</th>
                  <th className="px-4 py-2.5 text-left font-semibold text-slate-700">Sensor / Source</th>
                  <th className="px-4 py-2.5 text-right font-semibold text-slate-700">Extracted Value</th>
                  <th className="px-4 py-2.5 text-left font-semibold text-slate-700">Unit</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {featureKeys.map((key) => {
                  const item = summary[key];
                  return (
                    <tr key={key} className="hover:bg-slate-50/70">
                      <td className="px-4 py-2.5 font-medium text-slate-900">
                        {item.name}
                        <span className="block text-[10px] text-slate-400 font-mono">{key}</span>
                      </td>
                      <td className="px-4 py-2.5 text-slate-600">
                        <span className="inline-block bg-slate-100 px-2 py-0.5 rounded text-[11px] font-mono">
                          {item.source}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono font-bold text-slate-800">
                        {item.value !== null && item.value !== undefined ? item.value : '—'}
                      </td>
                      <td className="px-4 py-2.5 text-slate-500">{item.unit}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="text-[11px] text-slate-400 mt-3">
            * All features are extracted according to phenology-anchored grain filling windows and preprocessed via the trained SimpleImputer pipeline.
          </p>
        </div>
      )}
    </div>
  );
}
