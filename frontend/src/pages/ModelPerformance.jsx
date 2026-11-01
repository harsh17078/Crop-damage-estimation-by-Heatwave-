import React, { useState, useEffect } from 'react';
import { getModelPerformance } from '../api';
import { BarChart3, AlertTriangle, ShieldCheck, CheckCircle2, Info, Check } from 'lucide-react';

const CONFUSION_LABELS = ['NOT DAMAGED', 'LESS DAMAGED', 'MODERATELY', 'SEVERELY'];

export default function ModelPerformance() {
  const [performance, setPerformance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getModelPerformance()
      .then(res => setPerformance(res.performance))
      .finally(() => setLoading(false));
  }, []);

  if (loading || !performance) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 text-center text-slate-500">
        Loading model validation reports...
      </div>
    );
  }

  const { overall, per_class, confusion_matrix, confidence_calibration, limitations } = performance;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      <div>
        <h1 className="text-xl sm:text-2xl font-bold text-slate-900 flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-emerald-700" />
          Model Evaluation & Performance Diagnostics
        </h1>
        <p className="text-xs sm:text-sm text-slate-500 mt-1">
          Rigorous out-of-sample evaluation on the 2019–2024 rolling-origin holdout dataset (222 district-year test samples).
        </p>
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Test Accuracy</p>
          <p className="text-3xl font-black text-emerald-800 mt-1">{(overall.accuracy * 100).toFixed(2)}%</p>
          <p className="text-[11px] text-slate-400 mt-0.5">222 test samples</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">ROC-AUC (OvR)</p>
          <p className="text-3xl font-black text-teal-700 mt-1">{(overall.roc_auc_ovr * 100).toFixed(2)}%</p>
          <p className="text-[11px] text-slate-400 mt-0.5">One-vs-Rest discrimination</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Cohen's Kappa</p>
          <p className="text-3xl font-black text-blue-800 mt-1">{overall.cohen_kappa.toFixed(4)}</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Substantial agreement</p>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200 shadow-sm">
          <p className="text-xs font-semibold text-slate-500 uppercase">Sev+Mod → None FN</p>
          <p className="text-3xl font-black text-rose-600 mt-1">{(overall.sev_mod_fn_rate * 100).toFixed(2)}%</p>
          <p className="text-[11px] text-slate-400 mt-0.5">Damaged missed as healthy</p>
        </div>
      </div>

      {/* Two Column: Per-Class Table & Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Per-Class Metrics */}
        <div className="lg:col-span-7 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-3">
            Per-Class Validation Performance
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-xs">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-3.5 py-2.5 text-left font-semibold text-slate-700">Class Label</th>
                  <th className="px-3.5 py-2.5 text-right font-semibold text-slate-700">Support</th>
                  <th className="px-3.5 py-2.5 text-right font-semibold text-slate-700">Precision</th>
                  <th className="px-3.5 py-2.5 text-right font-semibold text-slate-700">Recall</th>
                  <th className="px-3.5 py-2.5 text-right font-semibold text-slate-700">F1-Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {per_class.map((c) => (
                  <tr key={c.class_id} className="hover:bg-slate-50">
                    <td className="px-3.5 py-2.5 font-medium flex items-center gap-2">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color }}></span>
                      <div>
                        <span className="font-bold text-slate-900">{c.frontend_label}</span>
                        <span className="block text-[10px] text-slate-400">{c.internal_label}</span>
                      </div>
                    </td>
                    <td className="px-3.5 py-2.5 text-right font-mono text-slate-700">{c.support}</td>
                    <td className="px-3.5 py-2.5 text-right font-mono font-bold text-slate-800">
                      {(c.precision * 100).toFixed(1)}%
                    </td>
                    <td className="px-3.5 py-2.5 text-right font-mono font-bold text-slate-800">
                      {(c.recall * 100).toFixed(1)}%
                    </td>
                    <td className="px-3.5 py-2.5 text-right font-mono font-bold text-emerald-800">
                      {c.f1_score.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[11px] text-slate-400 pt-1">
            * Holdout test period: 2019 to 2024 across 37 districts (6 unseen harvest years).
          </div>
        </div>

        {/* 4x4 Confusion Matrix */}
        <div className="lg:col-span-5 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <div className="border-b border-slate-100 pb-3 flex items-center justify-between">
            <h2 className="text-base font-bold text-slate-900">
              Holdout Confusion Matrix
            </h2>
            <span className="text-[11px] text-slate-400">Rows: True | Cols: Pred</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-center text-xs">
              <thead>
                <tr>
                  <th className="p-1"></th>
                  {CONFUSION_LABELS.map((col, j) => (
                    <th key={j} className="p-2 text-[10px] font-bold text-slate-600 uppercase tracking-tighter">
                      Pred {j}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusion_matrix.map((row, i) => (
                  <tr key={i}>
                    <td className="p-2 text-left text-[10px] font-bold text-slate-600 whitespace-nowrap">
                      True {i} ({CONFUSION_LABELS[i]})
                    </td>
                    {row.map((val, j) => {
                      const isDiag = i === j;
                      const bg = isDiag
                        ? val > 50 ? 'bg-emerald-600 text-white font-bold' : 'bg-emerald-100 text-emerald-900 font-bold'
                        : val > 10 ? 'bg-amber-100 text-amber-900 font-semibold' : 'bg-slate-50 text-slate-600';
                      return (
                        <td key={j} className="p-1">
                          <div className={`p-2.5 rounded-lg font-mono text-xs ${bg}`}>
                            {val}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="text-[11px] text-slate-400 pt-1">
            Diagonal cells represent correct classifications. Severe damage predictions focus on avoiding false negatives.
          </div>
        </div>
      </div>

      {/* Confidence Calibration & Model Limitations */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Confidence Calibration */}
        <div className="lg:col-span-5 bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
          <h2 className="text-base font-bold text-slate-900 border-b border-slate-100 pb-3">
            Confidence Calibration Analysis
          </h2>
          <div className="space-y-3">
            {confidence_calibration.map((cal, i) => (
              <div key={i} className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 flex items-center justify-between">
                <div>
                  <span className="font-bold text-xs text-slate-900">{cal.level} CONFIDENCE</span>
                  <p className="text-[11px] text-slate-400 mt-0.5">Threshold: {cal.min_prob}</p>
                </div>
                <div className="text-right">
                  <span className="text-base font-extrabold font-mono text-emerald-800">
                    {(cal.accuracy * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-slate-400 block">{cal.samples} samples</span>
                </div>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-slate-400">
            Predictions flagged with HIGH confidence achieve 95.0% accuracy on unseen test years.
          </p>
        </div>

        {/* Limitations & Academic Integrity */}
        <div className="lg:col-span-7 bg-amber-50/60 p-6 rounded-2xl border border-amber-200 shadow-sm space-y-4">
          <div className="border-b border-amber-200 pb-3 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-700" />
            <h2 className="text-base font-bold text-amber-950">
              Documented Limitations & Research Disclosure
            </h2>
          </div>
          <ul className="space-y-2.5 text-xs text-amber-950/85">
            {limitations.map((lim, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className="text-amber-600 font-bold">•</span>
                <span className="leading-relaxed">{lim}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
