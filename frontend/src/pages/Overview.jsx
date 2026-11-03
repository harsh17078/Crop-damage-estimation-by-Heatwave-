import React from 'react';
import {
  ArrowRight, ShieldCheck, Database, Layers, Satellite,
  CheckCircle2, Cpu, Globe,
  Zap, Clock, Shield, TrendingDown, Users, FileText, Check, DollarSign
} from 'lucide-react';

export default function Overview({ onStartAnalysis }) {
  const metrics = [
    { label: 'Assessment Speed', value: '< 5 sec', desc: 'Satellite ingestion to verdict' },
    { label: 'Field Cost Savings', value: 'Up to 80%', desc: 'vs. manual ground surveys' },
    { label: 'Coverage Scale', value: '37 Districts', desc: 'Punjab, Haryana & Western UP' },
    { label: 'Historical Archive', value: '22 Years', desc: '2003–2024 continuous data' },
  ];

  const pipelineSteps = [
    {
      step: '01',
      title: 'Pin Location',
      desc: 'Select district or drop a pin on the interactive map. Zero field survey hardware needed.',
      icon: Globe,
    },
    {
      step: '02',
      title: 'Instant Satellite Scan',
      desc: 'Direct Google Earth Engine queries across MODIS (thermal/vegetation), ERA5-Land, and CHIRPS rainfall.',
      icon: Satellite,
    },
    {
      step: '03',
      title: 'Canopy & Climate Profiling',
      desc: 'Extracts 89 phenology-anchored stress indicators capturing canopy desiccation and heatwave exposure.',
      icon: Layers,
    },
    {
      step: '04',
      title: 'Machine Learning Verdict',
      desc: 'Trained ExtraTrees model classifies damage tier (Healthy to Severe) with confidence calibration.',
      icon: Cpu,
    },
    {
      step: '05',
      title: 'Auditable Quality Check',
      desc: 'Microclimate anomaly flags identify rare weather patterns; instant one-click executive report export.',
      icon: ShieldCheck,
    },
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
    <div className="space-y-8 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8">

      {/* Executive Summary Banner */}
      <div className="bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-400 text-amber-950 rounded-2xl p-5 sm:p-7 shadow-lg border border-amber-300 relative overflow-hidden">
        <div className="absolute -right-16 -top-16 w-64 h-64 bg-white/20 rounded-full blur-2xl pointer-events-none" />
        <div className="relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="inline-flex items-center space-x-2 bg-amber-900/10 border border-amber-900/20 text-amber-900 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
              <Zap className="w-3.5 h-3.5" />
              <span>Executive Summary</span>
            </div>
            <h2 className="text-lg sm:text-xl lg:text-2xl font-black leading-tight max-w-2xl">
              Instant multi-spectral damage assessment across 37 districts in seconds — reducing manual crop inspection costs by up to 80%.
            </h2>
            <p className="text-xs sm:text-sm text-amber-900/80 max-w-2xl font-medium leading-relaxed">
              Empowering agricultural insurers, state disaster boards, and relief officers with tamper-proof satellite intelligence to fast-track genuine claims and eliminate assessment backlogs.
            </p>
          </div>
          <button
            onClick={onStartAnalysis}
            id="hero-cta-btn"
            className="flex-shrink-0 inline-flex items-center space-x-2 bg-amber-950 text-amber-100 hover:bg-black px-5 py-3 rounded-xl font-bold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98] whitespace-nowrap"
          >
            <span>Test Live Assessment</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Primary Hero Banner */}
      <div className="bg-gradient-to-br from-emerald-900 via-emerald-800 to-teal-950 text-white rounded-2xl p-6 sm:p-10 shadow-xl border border-emerald-700/50 relative overflow-hidden">
        <div className="absolute -right-20 -bottom-20 w-80 sm:w-96 h-80 sm:h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center space-x-2 bg-emerald-700/60 border border-emerald-500/40 text-emerald-200 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider">
            <span>Satellite Agrometeorological Intelligence</span>
          </div>
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight leading-tight">
            Satellite-Powered Wheat Crop Heatwave Risk & Damage Estimation
          </h1>
          <p className="text-sm sm:text-base text-emerald-100/90 leading-relaxed">
            Connecting Google Earth Engine with trained ensemble Machine Learning to detect terminal heat stress, canopy dehydration, and crop loss in seconds across{' '}
            <strong className="text-white">Punjab, Haryana, and Western Uttar Pradesh</strong>.
          </p>
          <div className="pt-1 flex flex-wrap gap-3">
            <button
              onClick={onStartAnalysis}
              id="main-cta-btn"
              className="inline-flex items-center space-x-2 bg-white text-emerald-900 hover:bg-emerald-50 px-5 py-3 rounded-xl font-bold text-sm shadow-md transition-all hover:scale-[1.02] active:scale-[0.98]"
            >
              <span>Launch Assessment Tool</span>
              <ArrowRight className="w-4 h-4" />
            </button>
            <div className="flex items-center space-x-2 text-xs text-emerald-200 px-3 py-2 bg-emerald-950/40 rounded-lg border border-emerald-800">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>37 Districts · 3 States · 2003–2024 GEE Archive</span>
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-5">
        {metrics.map((m, idx) => (
          <div key={idx} className="bg-white rounded-xl p-4 sm:p-5 border border-slate-200 shadow-sm hover:border-emerald-300 hover:shadow-md transition-all">
            <p className="text-[10px] sm:text-xs font-medium text-slate-500 uppercase tracking-wider">{m.label}</p>
            <p className="text-xl sm:text-2xl lg:text-3xl font-extrabold text-emerald-800 mt-1">{m.value}</p>
            <p className="text-[10px] sm:text-[11px] text-slate-400 mt-1">{m.desc}</p>
          </div>
        ))}
      </div>

      {/* Cost & Time Comparison Table */}
      <div className="bg-white rounded-2xl p-5 sm:p-7 border border-slate-200 shadow-sm space-y-5">
        <div className="border-b border-slate-100 pb-4">
          <div className="inline-flex items-center space-x-2 text-emerald-700 bg-emerald-50 px-2.5 py-1 rounded-full text-xs font-semibold mb-2">
            <TrendingDown className="w-3.5 h-3.5" />
            <span>Operational Efficiency</span>
          </div>
          <h2 className="text-base sm:text-lg font-bold text-slate-900">
            Manual Ground Inspection vs. Satellite AI Verification
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            Why agricultural insurers and government agencies are transitioning to remote-sensing automation.
          </p>
        </div>
        <div className="overflow-x-auto -mx-2 sm:mx-0">
          <table className="w-full text-left text-xs sm:text-sm min-w-[560px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-500 text-xs uppercase tracking-wider">
                <th className="py-3 px-3 sm:px-4 font-semibold">Evaluation Factor</th>
                <th className="py-3 px-3 sm:px-4 font-semibold text-rose-700 bg-rose-50/50">Traditional Ground Surveys</th>
                <th className="py-3 px-3 sm:px-4 font-semibold text-emerald-800 bg-emerald-50/50">Our GEE + ExtraTrees Solution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {comparisonData.map((row, i) => {
                const Icon = row.icon;
                return (
                  <tr key={i} className="hover:bg-slate-50/60 transition-colors">
                    <td className="py-3 px-3 sm:px-4 font-medium text-slate-900">
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                        <span>{row.dimension}</span>
                      </div>
                    </td>
                    <td className="py-3 px-3 sm:px-4 bg-rose-50/20">
                      <span className="text-rose-700 font-semibold">{row.traditional}</span>
                    </td>
                    <td className="py-3 px-3 sm:px-4 font-bold text-emerald-800 bg-emerald-50/20">
                      <span className="flex items-center gap-1.5">
                        <Check className="w-3.5 h-3.5 text-emerald-600 flex-shrink-0" />
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

      {/* 5-Step Pipeline Architecture */}
      <div className="bg-white rounded-2xl p-5 sm:p-7 border border-slate-200 shadow-sm space-y-5">
        <div className="border-b border-slate-100 pb-4">
          <h2 className="text-base sm:text-lg font-bold text-slate-900 flex items-center gap-2">
            <Layers className="w-5 h-5 text-emerald-700" />
            The 5-Second Verification Journey
          </h2>
          <p className="text-xs sm:text-sm text-slate-500 mt-1">
            From geographic coordinate to risk classification in five automated stages.
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4">
          {pipelineSteps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={idx}
                className="relative bg-slate-50 p-4 rounded-xl border border-slate-200/80 hover:bg-emerald-50/50 hover:border-emerald-200 transition-all duration-150"
              >
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded">
                    {step.step}
                  </span>
                  <Icon className="w-4 h-4 text-slate-400" />
                </div>
                <h3 className="font-semibold text-sm text-slate-900">{step.title}</h3>
                <p className="text-xs text-slate-600 mt-1.5 leading-relaxed">{step.desc}</p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Bottom two-column: Coverage & Trust */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 sm:gap-6">
        {/* Coverage */}
        <div className="bg-white rounded-2xl p-5 sm:p-6 border border-slate-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Database className="w-5 h-5 text-emerald-700" />
            Enterprise Geographic Coverage
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-slate-600">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">✓</span>
              <span><strong>3 States:</strong> Punjab (13 districts), Haryana (14 districts), Uttar Pradesh (10 districts).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">✓</span>
              <span><strong>NASA MODIS:</strong> MOD13Q1 (250m NDVI/EVI), MOD09A1 (500m NDMI), MOD11A2 (1km Day/Night LST).</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">✓</span>
              <span><strong>ECMWF ERA5-Land & CHIRPS:</strong> High-resolution daily air temperature, soil moisture, VPD, and rainfall.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">✓</span>
              <span><strong>Phenology Anchoring:</strong> Dynamic stage alignment relative to peak vegetative greenness for biological precision.</span>
            </li>
          </ul>
        </div>

        {/* Enterprise Trust */}
        <div className="bg-emerald-50/60 rounded-2xl p-5 sm:p-6 border border-emerald-200 shadow-sm space-y-4">
          <h3 className="text-base font-bold text-emerald-950 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-emerald-700" />
            Enterprise Trust & Reliability
          </h3>
          <ul className="space-y-2.5 text-xs sm:text-sm text-emerald-900/80">
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">•</span>
              <span><strong>Zero Blind Faith:</strong> High-confidence predictions achieve <strong>95% accuracy</strong> on unseen historical test years.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">•</span>
              <span><strong>Tamper-Proof Audit:</strong> Every query produces an exportable digital record linking coordinates, satellite bands, and predictions.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">•</span>
              <span><strong>Anomaly Safeguards:</strong> Unusual microclimates are flagged for human validation rather than returning overconfident outputs.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-emerald-600 font-bold mt-0.5">•</span>
              <span><strong>Rolling Multi-Year Validation:</strong> Trained on 2003–2018 and independently validated on 2019–2024 holdout seasons.</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
