import React from 'react';

const CLASS_CONFIG = [
  {
    id: 0,
    key: 'not_damaged',
    label: 'NOT DAMAGED',
    pitchLabel: 'Healthy',
    scientific: 'No Significant Heatwave Damage',
    color: '#16a34a',
    bg: 'bg-emerald-500',
    lightBg: 'bg-emerald-50',
    border: 'border-emerald-300',
    badge: 'bg-emerald-100 text-emerald-800'
  },
  {
    id: 1,
    key: 'less_damaged',
    label: 'LESS DAMAGED',
    pitchLabel: 'Low Risk',
    scientific: 'Mild Heatwave Damage',
    color: '#eab308',
    bg: 'bg-amber-500',
    lightBg: 'bg-amber-50',
    border: 'border-amber-300',
    badge: 'bg-amber-100 text-amber-800'
  },
  {
    id: 2,
    key: 'moderately_damaged',
    label: 'MODERATELY DAMAGED',
    pitchLabel: 'Medium Risk',
    scientific: 'Moderate Heatwave Damage',
    color: '#f97316',
    bg: 'bg-orange-500',
    lightBg: 'bg-orange-50',
    border: 'border-orange-300',
    badge: 'bg-orange-100 text-orange-800'
  },
  {
    id: 3,
    key: 'severely_damaged',
    label: 'SEVERELY DAMAGED',
    pitchLabel: 'High Risk',
    scientific: 'Severe Heatwave Damage',
    color: '#dc2626',
    bg: 'bg-rose-600',
    lightBg: 'bg-rose-50',
    border: 'border-rose-300',
    badge: 'bg-rose-100 text-rose-800'
  }
];

export default function ProbabilityBar({ probabilities, predictedClassId, pitchMode }) {
  if (!probabilities) return null;

  return (
    <div className="space-y-3.5">
      {CLASS_CONFIG.map((cls) => {
        const prob = probabilities[cls.key] || 0;
        const pct = (prob * 100).toFixed(1);
        const isPredicted = predictedClassId === cls.id;

        return (
          <div
            key={cls.id}
            className={`p-2.5 rounded-lg border transition-all ${
              isPredicted ? `${cls.lightBg} ${cls.border} shadow-sm ring-1 ring-emerald-500/30` : 'bg-slate-50 border-slate-200/70'
            }`}
          >
            <div className="flex items-center justify-between text-xs sm:text-sm mb-1.5">
              <div className="flex items-center space-x-2">
                <span
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{ backgroundColor: cls.color }}
                />
                <span className={`font-semibold ${isPredicted ? 'text-slate-900' : 'text-slate-700'}`}>
                  {pitchMode ? cls.pitchLabel : cls.label}
                </span>
                {isPredicted && (
                  <span className="text-[10px] font-bold uppercase tracking-wider bg-emerald-600 text-white px-1.5 py-0.5 rounded">
                    {pitchMode ? 'Result' : 'Predicted Class'}
                  </span>
                )}
              </div>
              <span className="font-mono font-bold text-slate-800">{pct}%</span>
            </div>

            {/* Progress bar */}
            <div className="w-full h-2.5 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={`h-full ${cls.bg} rounded-full transition-all duration-700`}
                style={{ width: `${Math.max(prob * 100, 2)}%` }}
              />
            </div>

            {!pitchMode && (
              <div className="flex justify-between items-center mt-1 text-[11px] text-slate-400">
                <span>{cls.scientific}</span>
                <span>p = {prob.toFixed(4)}</span>
              </div>
            )}
          </div>
        );
      })}

      {!pitchMode && (
        <p className="text-[11px] text-slate-500 italic text-center mt-2">
          Note: These scores represent ExtraTrees model output probabilities across 89 satellite & meteorological features.
        </p>
      )}
    </div>
  );
}
