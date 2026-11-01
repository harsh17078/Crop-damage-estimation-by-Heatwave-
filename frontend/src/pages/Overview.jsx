import React from 'react';
import { 
  ArrowRight, ShieldCheck, Database, Layers, Satellite, 
  Thermometer, CheckCircle2, AlertTriangle, Cpu, BarChart2, Globe,
  Zap, Clock, Shield, TrendingDown, Users, FileText, Check, DollarSign
} from 'lucide-react';

export default function Overview({ onStartAnalysis, pitchMode }) {
  const metrics = pitchMode ? [
    { label: 'Assessment Speed', value: '< 5 sec', desc: 'Satellite ingestion to verdict' },
    { label: 'Field Cost Savings', value: 'Up to 80%', desc: 'vs. manual ground surveys' },
    { label: 'Coverage Scale', value: '37 Districts', desc: 'Punjab, Haryana & Western UP' },
    { label: 'Historical Archive', value: '22 Years', desc: '2003–2024 continuous data' },
  ] : [
    { label: 'Holdout Accuracy', value: '76.13%', desc: '2019–2024 rolling holdout' },
    { label: 'ROC-AUC (OvR)', value: '90.59%', desc: 'Multi-class discrimination' },
    { label: 'Macro F1-Score', value: '54.59%', desc: 'Balanced 4-class evaluation' },
    { label: 'Sev+Mod → None FN', value: '3.92%', desc: 'Minimal critical misclassifications' },
  ];

  const pipelineSteps = [
    {
      step: '01',
      title: pitchMode ? 'Pin Location' : 'Geographic Selection',
      desc: pitchMode 
        ? 'Select district or drop a pin on the interactive map. Zero field survey hardware needed.'
        : 'User selects State, District, Harvest Year, and precise coordinates via interactive Leaflet map.',
      icon: Globe
    },
    {
      step: '02',
      title: pitchMode ? 'Instant Satellite Scan' : 'GEE Remote Sensing',
      desc: pitchMode
        ? 'Direct Google Earth Engine queries across MODIS (thermal/vegetation), ERA5-Land, and CHIRPS rainfall.'
        : 'Google Earth Engine queries MODIS (NDVI, EVI, NDMI, LST), ERA5-Land (Tmax, SM, VPD), and CHIRPS precipitation.',
      icon: Satellite
    },
    {
      step: '03',
      title: pitchMode ? 'Canopy & Climate Profiling' : 'Feature Engineering',
      desc: pitchMode
        ? 'Extracts 89 phenology-anchored stress indicators capturing canopy desiccation and heatwave exposure.'
        : '89 phenology-anchored features & domain interaction terms (thermal-radiative stress, canopy senescence).',
      icon: Layers
    },
    {
      step: '04',
      title: pitchMode ? 'Machine Learning Verdict' : 'ML Classification',
      desc: pitchMode
        ? 'Trained ExtraTrees model classifies damage tier (Healthy to Severe) with confidence calibration.'
        : 'Trained ExtraTreesClassifier (Balanced) predicts crop damage severity with confidence calibration.',
      icon: Cpu
    },
    {
      step: '05',
      title: pitchMode ? 'Auditable Quality Check' : 'Safety & Verification',
      desc: pitchMode
        ? 'Microclimate anomaly flags identify rare weather patterns; instant one-click executive report export.'
        : 'PCA-Mahalanobis OOD detection (threshold 53.72) and physical consistency rule verification layer.',
      icon: ShieldCheck
    }
  ];

  const comparisonData = [
    { dimension: 'Turnaround Time', traditional: '2 to 4 Weeks', satellite: 'Under 5 Seconds', icon: Clock },
    { dimension: 'Operational Cost', traditional: 'High (Labor, Travel, Manual Logging)', satellite: '80% Lower (Cloud-native)', icon: DollarSign },
    { dimension: 'Human Bias & Dispute Risk', traditional: 'High (Subjective visual estimation)', satellite: 'Objective & Reproducible', icon: Shield },
    { dimension: 'Coverage Capacity', traditional: 'Limited to sample plots', satellite: 'Wall-to-wall district coverage', icon: Users },
    { dimension: 'Multi-Year Historical Depth', traditional: 'Fragmented records', satellite: '22-Year satellite baseline', icon: Database },
    { dimension: 'Audit & Claim Verification', traditional: 'Paper receipts & manual claims', satellite: 'Digital timestamps & CSV/PDF audit', icon: FileText },
  ];

  return (
    <div className="space-y-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Pitch Mode Hero Hook Banner (Above the Fold) */}
      {pitchMode && (
        <div className="bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-400 text-amber-950 rounded-2xl p-6 sm:p-8 shadow-lg border border-amber-300 relative overflow-hidden">
          <div className="absolute -right-16 -top-16 w-64 h-64 bg-white/20 rounded-full blur-2xl pointer-events-none" />
          <div className="relative z-10 flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="space-y-2 text-center md:text-left">
              <div className="inline-flex items-center space-x-2 bg-amber-900/10 border border-amber-900/20 text-amber-900 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                <Zap className="w-3.5 h-3.5" />
                <span>Executive Summary & Pitch Value</span>
              </div>
              <h2 className="text-xl sm:text-2xl lg:text-3xl font-black leading-tight">
                Instant multi-spectral damage assessment across 37 districts in seconds — reducing manual crop inspection costs by up to 80%.
              </h2>
              <p className="text-xs sm:text-sm text-amber-900/90 max-w-2xl font-medium leading-relaxed">
                Empowering agricultural insurers, state disaster boards, and relief officers with tamper-proof satellite intelligence to fast-track genuine claims and eliminate assessment backlogs.
              </p>
            </div>
            <button
              onClick={onStartAnalysis}
              className="flex-shrink-0 inline-flex items-center space-x-2 bg-amber-950 text-amber-100 hover:bg-black px-6 py-3.5 rounded-xl font-bold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>Test Live Assessment</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Primary Banner */}
      <div className="bg-gradient-to-br from-emerald-900 via-emerald-800 to-teal-950 text-white rounded-2xl p-8 sm:p-12 shadow-xl border border-emerald-700/50 relative overflow-hidden">
        <div className="absolute -right-20 -bottom-20 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-5">
          <div className="inline-flex items-center space-x-2 bg-emerald-700/60 border border-emerald-500/40 text-emerald-200 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">
            <span>{pitchMode ? 'Satellite Agrometeorological Intelligence' : 'Research-Grade Agricultural Pipeline'}</span>
          </div>

          <h1 className="text-2xl sm:text-4xl font-extrabold tracking-tight leading-tight">
            {pitchMode 
              ? 'Satellite-Powered Wheat Crop Heatwave Risk & Damage Estimation'
              : 'Machine Learning-Based Wheat Crop Damage Estimation Due to Heat Waves'}
          </h1>

          <p className="text-sm sm:text-base text-emerald-100/90 leading-relaxed">
            {pitchMode ? (
              <>
                Connecting Google Earth Engine with trained ensemble Machine Learning to detect terminal heat stress, canopy dehydration, and crop loss in seconds across <strong className="text-white">Punjab, Haryana, and Western Uttar Pradesh</strong>.
              </>
            ) : (
              <>
                Connecting an ensemble <strong className="text-white">ExtraTreesClassifier</strong> with{' '}
                <strong className="text-white">Google Earth Engine (GEE)</strong> to extract multi-source satellite 
                (MODIS), reanalysis weather (ERA5-Land), and rainfall (CHIRPS) observations for location-based 
                damage risk assessment across the Indo-Gangetic Wheat Belt.
              </>
            )}
          </p>

          <div className="pt-2 flex flex-wrap gap-4">
            <button
              onClick={onStartAnalysis}
              className="inline-flex items-center space-x-2 bg-white text-emerald-900 hover:bg-emerald-50 px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>{pitchMode ? 'Launch Assessment Tool' : 'Start Crop Damage Analysis'}</span>
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

      {/* Cost & Time Comparison Narrative — Pitch Mode */}
      {pitchMode && (
        <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
          <div className="border-b border-slate-100 pb-4">
            <div className="inline-flex items-center space-x-2 text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full text-xs font-semibold mb-2">
              <TrendingDown className="w-3.5 h-3.5" />
              <span>Operational Efficiency</span>
            </div>
            <h2 className="text-lg sm:text-xl font-bold text-slate-900">
              Manual Ground Inspection vs. Satellite AI Verification
            </h2>
            <p className="text-xs sm:text-sm text-slate-500 mt-1">
              Why agricultural insurers and government agencies are transitioning to remote-sensing automation.
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 text-xs uppercase tracking-wider">
                  <th className="py-3 px-4 font-semibold">Evaluation Factor</th>
                  <th className="py-3 px-4 font-semibold text-rose-700 bg-rose-50/50 rounded-tl-lg">Traditional Ground Surveys</th>
                  <th className="py-3 px-4 font-semibold text-emerald-800 bg-emerald-50/50 rounded-tr-lg">Our GEE + ExtraTrees Solution</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {comparisonData.map((row, i) => {
                  const Icon = row.icon;
                  return (
                    <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                      <td className="py-3.5 px-4 font-medium text-slate-900 flex items-center gap-2.5">
                        <Icon className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                        <span>{row.dimension}</span>
                      </td>
                      <td className="py-3.5 px-4 text-slate-600 bg-rose-50/20 font-medium">
                        <span className="text-rose-700 font-semibold">{row.traditional}</span>
                      </td>
                      <td className="py-3.5 px-4 font-bold text-emerald-800 bg-emerald-50/20">
                        <span className="flex items-center gap-1.5">
                          <Check className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                          <span>{row.satellite}</span>
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* End-to-End Workflow Architecture */}
      <div className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="border-b border-slate-100 pb-4">
          <h2 className="text-lg sm:text-xl font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-700" />
            {pitchMode ? 'The 5-Second Verification Journey' : 'End-to-End System Architecture'}
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            {pitchMode
              ? 'From geographic coordinate to risk classification in five automated stages.'
              : 'Rigorous satellite data retrieval, phenological windowing, and machine learning inference workflow.'}
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

      {/* Two Column Section: Project Scope & Trust */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Project Scope */}
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-700" />
            {pitchMode ? 'Enterprise Geographic Coverage' : 'Coverage & Data Integration'}
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>3 States:</strong> Punjab (13 districts), Haryana (14 districts), Uttar Pradesh (10 districts).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>NASA MODIS:</strong> MOD13Q1 (250m NDVI/EVI), MOD09A1 (500m NDMI), MOD11A2 (1km Day/Night LST).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>ECMWF ERA5-Land & CHIRPS:</strong> High-resolution daily air temperature, soil moisture, VPD, and rainfall.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">✓</span>
              <span><strong>Phenology Anchoring:</strong> Dynamic stage alignment relative to peak vegetative greenness ($P$) for biological precision.</span>
            </li>
          </ul>
        </div>

        {/* Scientific Integrity & Trust */}
        <div className="bg-emerald-50/50 rounded-2xl p-6 border border-emerald-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-emerald-950 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-700" />
            {pitchMode ? 'Enterprise Trust & Reliability' : 'Scientific Integrity & Model Limitations'}
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-emerald-900/80">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">•</span>
              <span><strong>Zero Blind Faith:</strong> High-confidence predictions achieve <strong>95% accuracy</strong> on unseen historical test years.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">•</span>
              <span><strong>Tamper-Proof Audit:</strong> Every query produces an exportable digital record linking coordinates, satellite bands, and predictions.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">•</span>
              <span><strong>Anomaly Safeguards:</strong> Unusual microclimates are flagged for human validation rather than returning overconfident outputs.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold">•</span>
              <span><strong>Rolling Multi-Year Validation:</strong> Trained on 2003–2018 and independently validated on 2019–2024 holdout seasons.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
