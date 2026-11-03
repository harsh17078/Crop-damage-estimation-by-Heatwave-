import React from 'react';

const CLASS_CONFIG = [
  {
    id: 0,
    key: 'not_damaged',
    label: 'Healthy',
    color: '#16a34a',
    bg: 'bg-emerald-500',
    lightBg: 'bg-emerald-50',
    border: 'border-emerald-300',
  },
  {
    id: 1,
    key: 'less_damaged',
    label: 'Low Risk',
    color: '#eab308',
    bg: 'bg-amber-500',
    lightBg: 'bg-amber-50',
    border: 'border-amber-300',
  },
  {
    id: 2,
    key: 'moderately_damaged',
    label: 'Medium Risk',
    color: '#f97316',
    bg: 'bg-orange-500',
    lightBg: 'bg-orange-50',
    border: 'border-orange-300',
  },
  {
    id: 3,
    key: 'severely_damaged',
    label: 'High Risk',
    color: '#dc2626',
    bg: 'bg-rose-600',
    lightBg: 'bg-rose-50',
    border: 'border-rose-300',
  },
];

export default function ProbabilityBar({ probabilities, predictedClassId }) {
  if (!probabilities) return null;

  return (
    <div className="space-y-3">
      {CLASS_CONFIG.map((cls) => {
        const prob = probabilities[cls.key] || 0;
        const pct = (prob * 100).toFixed(1);
        const isPredicted = predictedClassId === cls.id;

        return (
          <div
            key={cls.id}
            className={`p-2.5 rounded-lg border transition-all ${
              isPredicted
                ? `${cls.lightBg} ${cls.border} shadow-sm ring-1 ring-emerald-500/30`
                : 'bg-slate-50 border-slate-200/70'
            }`}
          >
            <div className="flex items-center justify-between text-xs mb-1.5 gap-2">
              <div className="flex items-center space-x-2 min-w-0">
                <span
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ backgroundColor: cls.color }}
                />
                <span className={`font-semibold truncate ${isPredicted ? 'text-slate-900' : 'text-slate-700'}`}>
                  {cls.label}
                </span>
                {isPredicted && (
                  <span className="text-[9px] font-bold uppercase tracking-wider bg-emerald-600 text-white px-1.5 py-0.5 rounded flex-shrink-0">
                    Result
                  </span>
                )}
              </div>
              <span className="font-mono font-bold text-slate-800 flex-shrink-0">{pct}%</span>
            </div>

            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div
                className={`h-full ${cls.bg} rounded-full transition-all duration-700`}
                style={{ width: `${Math.max(prob * 100, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
