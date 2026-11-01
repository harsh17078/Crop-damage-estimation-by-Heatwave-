import React from 'react';
import { 
  ArrowRight, ShieldCheck, Database, Layers, Satellite, 
  Thermometer, CheckCircle2, AlertTriangle, Cpu, BarChart2, Globe
} from 'lucide-react';

export default function Overview({ onStartAnalysis }) {
  const metrics = [
    { label: 'Holdout Accuracy', value: '76.13%', desc: '2019–2024 rolling holdout' },
    { label: 'ROC-AUC (OvR)', value: '90.59%', desc: 'Multi-class discrimination' },
    { label: 'Macro F1-Score', value: '54.59%', desc: 'Balanced 4-class evaluation' },
    { label: 'Sev+Mod → None FN', value: '3.92%', desc: 'Minimal critical misclassifications' },
  ];

  const pipelineSteps = [
    {
      step: '01',
      title: 'Geographic Selection',
      desc: 'User selects State, District, Harvest Year, and precise coordinates via interactive Leaflet map.',
      icon: Globe
    },
    {
      step: '02',
      title: 'GEE Remote Sensing',
      desc: 'Google Earth Engine queries MODIS (NDVI, EVI, NDMI, LST), ERA5-Land (Tmax, SM, VPD), and CHIRPS precipitation.',
      icon: Satellite
    },
    {
      step: '03',
      title: 'Feature Engineering',
      desc: '89 phenology-anchored features & domain interaction terms (thermal-radiative stress, canopy senescence).',
      icon: Layers
    },
    {
      step: '04',
      title: 'ML Classification',
      desc: 'Trained ExtraTreesClassifier (Balanced) predicts crop damage severity with confidence calibration.',
      icon: Cpu
    },
    {
      step: '05',
      title: 'Safety & Verification',
      desc: 'PCA-Mahalanobis OOD detection (threshold 53.72) and physical consistency rule verification layer.',
      icon: ShieldCheck
    }
  ];

  return (
    <div className="space-y-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Banner */}
      <div className="bg-gradient-to-br from-emerald-900 via-emerald-800 to-teal-950 text-white rounded-2xl p-8 sm:p-12 shadow-xl border border-emerald-700/50 relative overflow-hidden">
        <div className="absolute -right-20 -bottom-20 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-5">
          <div className="inline-flex items-center space-x-2 bg-emerald-700/60 border border-emerald-500/40 text-emerald-200 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">
            <span>Research-Grade Agricultural Intelligence</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight leading-tight">
            Machine Learning-Based Wheat Crop Damage Estimation Due to Heat Waves
          </h1>

          <p className="text-sm sm:text-base text-emerald-100/90 leading-relaxed">
            Connecting an ensemble <strong className="text-white">ExtraTreesClassifier</strong> with{' '}
            <strong className="text-white">Google Earth Engine (GEE)</strong> to extract multi-source satellite 
            (MODIS), reanalysis weather (ERA5-Land), and rainfall (CHIRPS) observations for location-based 
            damage risk assessment across the Indo-Gangetic Wheat Belt.
          </p>

          <div className="pt-2 flex flex-wrap gap-4">
            <button
              onClick={onStartAnalysis}
              className="inline-flex items-center space-x-2 bg-white text-emerald-900 hover:bg-emerald-50 px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>Start Crop Damage Analysis</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex items-center space-x-2 text-xs text-emerald-200 px-3 py-2 bg-emerald-950/40 rounded-lg border border-emerald-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>37 Districts • 3 States • 2003–2024 GEE Archive</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        {metrics.map((m, idx) => (
          <div key={idx} className="bg-white rounded-xl p-5 border border-slate-200 shadow-sm hover:border-emerald-300 transition-all">
            <p className="text-xs font-medium text-slate-500 uppercase tracking-wider">{m.label}</p>
            <p className="text-2xl sm:text-3xl font-extrabold text-emerald-800 mt-1">{m.value}</p>
            <p className="text-[11px] text-slate-400 mt-1">{m.desc}</p>
          </div>
        ))}
      </div>

      {/* End-to-End Workflow Architecture */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="border-b border-slate-100 pb-4">
          <h2 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-700" />
            End-to-End System Architecture
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Rigorous satellite data retrieval, phenological windowing, and machine learning inference workflow.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {pipelineSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div key={idx} className="relative bg-slate-50 p-4 rounded-xl border border-slate-200/80 hover:bg-emerald-50/50 hover:border-emerald-200 transition-colors">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded">
                    {step.step}
                  </span>
                  <Icon className="w-4 h-4 text-slate-500" />
                </div>
                <h3 className="font-semibold text-sm text-slate-900">{step.title}</h3>
                <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Two Column Section: Project Scope & Scientific Honesty */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Project Scope */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-700" />
            Coverage & Data Integration
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>3 States:</strong> Punjab (13 districts), Haryana (14 districts), Uttar Pradesh (10 districts).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>MODIS:</strong> MOD13Q1 (250m NDVI/EVI), MOD09A1 (500m NDMI), MOD11A2 (1km Day/Night LST).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>ERA5-Land & CHIRPS:</strong> Daily maximum/minimum air temperature, soil moisture, VPD, and precipitation.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>Phenology Anchoring:</strong> Dynamic stage alignment relative to peak vegetative greenness ($P$).</span>
            </li>
          </ul>
        </div>

        {/* Scientific Integrity */}
        <div className="bg-emerald-50/50 rounded-2xl p-6 border border-emerald-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-emerald-950 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-600" />
            Scientific Integrity & Model Limitations
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-emerald-900/80">
            <li className="flex items-start gap-2">
              <span className="text-amber-600 font-bold">•</span>
              <span>Validation metrics (76.13% Acc, 90.59% ROC-AUC) reflect the 2019–2024 holdout test set, not live guarantees.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 font-bold">•</span>
              <span>Severe damage recall is 30.00% (6 of 20 holdout cases). Low overall severe recall is transparently flagged.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 font-bold">•</span>
              <span>NDVI drop alone does not prove heat damage; non-thermal causes (hail, disease, cold) trigger verification warnings.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-amber-600 font-bold">•</span>
              <span>Out-of-Distribution (OOD) detection warns if environmental features diverge from the 2003–2018 training space.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
