import React, { useState } from 'react';
import { HolographicHUD } from './components/HolographicHUD';
import { ChatStream } from './components/ChatStream';
import { RiskRadar } from './components/RiskRadar';
import { ActionButtons } from './components/ActionButtons';
import { GlassCard } from './components/ui/GlassCard';
import { ShieldCheck, Play, Terminal, Cpu, Trash2 } from 'lucide-react';

function App() {
  const [inputText, setInputText] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statelessMode, setStatelessMode] = useState(true); // Default to True (Test Mode)

  const [suspectName, setSuspectName] = useState('');
  const [contextFactors, setContextFactors] = useState([]);
  const [selectedResultSuspect, setSelectedResultSuspect] = useState('All');

  const toggleFactor = (factor) => {
    setContextFactors(prev =>
      prev.includes(factor)
        ? prev.filter(f => f !== factor)
        : [...prev, factor]
    );
  };

  const handleAnalyze = async () => {
    if (!inputText.trim()) return;
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          suspect_name: suspectName,
          stateless: statelessMode,
          context_factors: contextFactors
        }),
      });
      const result = await response.json();
      setData(result);
      // Auto-select the suspect from input if possible, or All
      // If the parser found "suspect" sender, we could default to them.
      // For now, default to All effectively shows Global
      setSelectedResultSuspect('All');
    } catch (error) {
      console.error("API Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // --- CLIENT-SIDE AGGREGATION (Phase 10) ---
  // ---------------------------------------------------------------------------
  // ROBUST METRICS CALCULATION (Phase 15 Fix)
  // ---------------------------------------------------------------------------
  const computedMetrics = React.useMemo(() => {
    if (!data || !data.segments) return null;

    const allSegments = data.segments;
    const senders = [...new Set(allSegments.map(s => s.sender_name))];

    // STEP 1: CREATE THE "ACTIVE" SUBSET
    // If "All" is selected, we look at everyone.
    // If a specific name is selected, we STRICTLY filter for that sender.
    const activeSegments = selectedResultSuspect === 'All'
      ? allSegments
      : allSegments.filter(s => s.sender_name === selectedResultSuspect);

    // EDGE CASE: Safety check if user has 0 messages
    if (!activeSegments || activeSegments.length === 0) {
      return {
        risk_score: 0,
        risk_level: "Safe",
        primary_pattern: "NONE",
        radar_chart_data: [],
        senders: ['All', ...senders],
        segments: [],
        darvo_score: 0,
        cycle_phase: data.cycle_phase || "Normal"
      };
    }

    // STEP 2: CALCULATE METRICS *ONLY* ON THE SUBSET
    const riskScores = activeSegments.map(s => s.risk_score);

    // Calculate Average on Subset
    const totalRisk = riskScores.reduce((sum, val) => sum + val, 0);
    const avgRisk = totalRisk / activeSegments.length;

    // Calculate Max Risk on Subset (Critical for finding threats)
    const maxRisk = Math.max(...riskScores);

    // STEP 3: APPLY "CRITICAL DOMINANCE" TO THE SUBSET
    let finalRiskInput = avgRisk;

    if (maxRisk > 0.85) {
      finalRiskInput = maxRisk; // Snap to Critical if THIS USER made a threat
    } else if (maxRisk > 0.60) {
      finalRiskInput = Math.max(avgRisk, 0.60); // Snap to Warning
    }

    // Determine Level
    let riskLevel = "Low";
    if (finalRiskInput > 0.8) riskLevel = "Critical";
    else if (finalRiskInput > 0.6) riskLevel = "High";
    else if (finalRiskInput > 0.3) riskLevel = "Medium";

    // 4. Calc Radar Data (Subset Only)
    const tacticMap = {
      'gaslighting': 'Gaslighting',
      'guilt_tripping': 'Guilt',
      'threatening_intimidation': 'Threats',
      'stonewalling': 'Silence',
      'love_bombing': 'Love Bomb',
      'deflection': 'Deflection'
    };

    const radarAgg = {
      'Gaslighting': 0, 'Guilt': 0, 'Threats': 0,
      'Silence': 0, 'Love Bomb': 0, 'Deflection': 0
    };

    activeSegments.forEach(seg => {
      if (!seg.tactic_scores) return;
      Object.entries(seg.tactic_scores).forEach(([key, val]) => {
        const radarKey = tacticMap[key];
        if (radarKey) {
          radarAgg[radarKey] = Math.max(radarAgg[radarKey], val);
        }
      });
    });

    const radarChartData = Object.entries(radarAgg).map(([subject, val]) => ({
      subject,
      A: Math.round(val * 100),
      fullMark: 100
    }));

    // Find Primary Pattern (with Priority Override)
    // Fix Phase 17: If Critical Risk, favor Threats > Gaslighting > Love Bomb
    const primaryPattern = Object.entries(radarAgg).reduce((a, b) => {
      // If scores are close (within 0.1), use priority list
      if (Math.abs(a[1] - b[1]) < 0.1) {
        const priority = { 'Threats': 3, 'Coercive Control': 3, 'Gaslighting': 2, 'Guilt': 2, 'Love Bomb': 1, 'Deflection': 1 };
        const pA = priority[a[0]] || 0;
        const pB = priority[b[0]] || 0;
        return pA >= pB ? a : b;
      }
      return a[1] > b[1] ? a : b;
    })[0] || "None";

    // 5. Calc DARVO (Subset Only)
    const maxDarvo = activeSegments.reduce((max, s) => Math.max(max, s.darvo_score || 0), 0);

    // 6. PHASE DETECTION OVERRIDE (Phase 16 Fix)
    // Problem: Global Phase (e.g., Explosion) leaks to Bystanders.
    // Fix: If Scoped Risk is SAFE, Force Phase to NORMAL.
    let localPhase = data.cycle_phase || "Normal";

    if (riskLevel === "Low") {
      localPhase = "Normal / Calm";
    } else if (riskLevel === "Medium") {
      // If global is Explosion, maybe show "Tension Building" for medium risk? 
      // Or just keep global if it's not 'Safe'.
      // For simplicity: If Low, Force Normal. Otherwise trust backend or Risk Level.
      if (localPhase === "scan_required") localPhase = "Tension Building";
    }

    return {
      risk_score: finalRiskInput,
      risk_level: riskLevel,
      primary_pattern: primaryPattern.toUpperCase(),
      radar_chart_data: radarChartData,
      senders: ['All', ...senders],
      segments: activeSegments,
      darvo_score: maxDarvo,
      cycle_phase: localPhase
    };


  }, [data, selectedResultSuspect]);

  const handleReset = async () => {
    if (!confirm("Are you sure you want to clear the session memory? This cannot be undone.")) return;
    try {
      await fetch('http://localhost:8000/api/reset', { method: 'POST' });
      setData(null);
      setInputText('');
      alert("Session memory cleared.");
    } catch (error) {
      console.error("Reset Error:", error);
    }
  };

  return (
    <div className="min-h-screen p-8 text-slate-200 font-inter">
      {/* Navbar */}
      <header className="flex justify-between items-center mb-10 max-w-[1600px] mx-auto">
        <h1 className="text-3xl font-orbitron font-bold glow-text flex items-center gap-3">
          <ShieldCheck className="w-8 h-8 text-neon-purple" />
          ManTacAi <span className="text-white text-xl opacity-80">PRO</span>
        </h1>
        <div className="flex items-center gap-4 text-xs font-mono text-neon-cyan/80">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-3 py-1 bg-red-500/10 border border-red-500/30 rounded text-red-400 hover:bg-red-500/20 transition-all font-orbitron"
          >
            <Trash2 className="w-4 h-4" /> RESET SESSION
          </button>
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 animate-pulse" /> SYSTEM ONLINE
          </div>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto grid grid-cols-12 gap-8">

        {/* LEFT COLUMN: Controls (3 cols) */}
        <div className="col-span-12 lg:col-span-3 space-y-6">
          <GlassCard className="h-full flex flex-col">
            <h2 className="font-orbitron text-lg text-white mb-4 flex items-center gap-2">
              <Terminal className="w-5 h-5 text-neon-purple" /> Input Data
            </h2>
            <textarea
              className="w-full h-[400px] bg-black/40 border border-glass-border rounded-lg p-4 text-sm font-mono text-slate-300 focus:border-neon-purple focus:outline-none resize-none mb-4 scrollbar-thin scrollbar-thumb-neon-purple/20"
              placeholder="Paste chat log here..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
            />

            {/* Suspect Name Input */}
            <div className="mb-4">
              <label className="text-xs font-orbitron text-slate-400 mb-1 block uppercase tracking-wide">TARGET SUSPECT (OPTIONAL)</label>
              <input
                type="text"
                value={suspectName}
                onChange={(e) => setSuspectName(e.target.value)}
                className="w-full bg-black/40 border border-glass-border rounded p-2 text-sm font-mono text-neon-red/80 focus:border-neon-red focus:outline-none placeholder-slate-600"
                placeholder="e.g. Alex_M"
              />
            </div>

            <button
              onClick={handleAnalyze}
              disabled={loading}
              className={`
                w-full py-4 rounded-lg font-orbitron font-bold tracking-wider transition-all
                flex items-center justify-center gap-2
                ${loading
                  ? 'bg-slate-700 cursor-not-allowed opacity-50'
                  : 'bg-neon-purple/10 border border-neon-purple text-neon-purple hover:bg-neon-purple/20 hover:shadow-[0_0_20px_rgba(168,85,247,0.3)]'
                }
              `}
            >
              {loading ? 'PROCESSING...' : <><Play className="w-4 h-4" /> INITIATE SCAN</>}
            </button>

            {/* Context Modifiers */}
            <div className="mt-4 border-t border-slate-700/50 pt-4">
              <h3 className="text-xs font-orbitron text-slate-400 mb-3 uppercase tracking-wide flex items-center gap-2">
                <ShieldCheck className="w-3 h-3 text-neon-cyan" /> Context Factors
              </h3>
              <div className="grid grid-cols-1 gap-2">
                {['financial_dependency', 'history_of_violence', 'isolation', 'stalking_history'].map(factor => (
                  <label key={factor} className="flex items-center gap-3 cursor-pointer group p-2 rounded hover:bg-slate-800/30 transition-all">
                    <div className="relative">
                      <input
                        type="checkbox"
                        className="peer sr-only"
                        checked={contextFactors.includes(factor)}
                        onChange={() => toggleFactor(factor)}
                      />
                      <div className="w-4 h-4 border border-slate-500 rounded bg-black/40 peer-checked:bg-neon-cyan peer-checked:border-neon-cyan transition-all shadow-[0_0_10px_rgba(6,186,212,0.2)] peer-checked:shadow-[0_0_10px_rgba(6,186,212,0.6)]"></div>
                    </div>
                    <span className={`text-xs font-mono capitalize transition-colors ${contextFactors.includes(factor) ? 'text-neon-cyan' : 'text-slate-400 group-hover:text-slate-200'}`}>
                      {factor.replace(/_/g, ' ')}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {/* Stateless Mode Toggle */}
            <label className="flex items-center justify-center gap-3 cursor-pointer group pt-2">
              <div className="relative">
                <input
                  type="checkbox"
                  className="sr-only peer"
                  checked={statelessMode}
                  onChange={(e) => setStatelessMode(e.target.checked)}
                />
                <div className="w-11 h-6 bg-slate-800 border border-slate-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-slate-400 after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-cyan/20 peer-checked:border-neon-cyan peer-checked:after:bg-neon-cyan peer-checked:after:border-neon-cyan"></div>
              </div>
              <span className={`font-orbitron text-xs transition-colors ${statelessMode ? 'text-neon-cyan' : 'text-slate-500'}`}>
                {statelessMode ? 'TEST MODE (STATELESS)' : 'SURVEILLANCE (STATEFUL)'}
              </span>
            </label>
          </GlassCard>
        </div>

        {/* RIGHT COLUMN: Output (9 cols) */}
        <div className="col-span-12 lg:col-span-9 space-y-6">

          {/* 1. SUSPECT SELECTOR (Phase 10) */}
          {computedMetrics && (
            <div className="flex justify-end mb-2">
              <select
                value={selectedResultSuspect}
                onChange={(e) => setSelectedResultSuspect(e.target.value)}
                className="bg-black/60 border border-neon-cyan/50 rounded px-4 py-2 text-neon-cyan font-orbitron text-sm focus:outline-none focus:border-neon-cyan"
              >
                {computedMetrics.senders.map(s => (
                  <option key={s} value={s} className="bg-slate-900 text-slate-200">{s === 'All' ? 'GLOBAL VIEW (ALL)' : `ANALYZE: ${s.toUpperCase()}`}</option>
                ))}
              </select>
            </div>
          )}

          {/* 2. HUD ROW */}
          <HolographicHUD metrics={computedMetrics || data} />

          {/* 2. VISUALIZATION ROW (Split 7/5) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8">
              {/* Show ALL segments in stream, but maybe highlight? 
                  Actually, if we filter the stream, it might lose context.
                  User said "The 'Global Score' is then just a math calculation".
                  Usually you want to see the whole chat, but metrics reflect the suspect.
                  Let's pass ALL segments to ChatStream so context is visible?
                  But computedMetrics.segments is filtered. 
                  Let's pass data.segments (ALL) to Stream, but maybe computedMetrics to Radar.
                  
                  Actually, user might want to see ONLY suspect messages?
                  "User can toggle between... and see Risk Level flip".
                  Usually a chat stream shows the conversation.
                  Let's keep Stream as GLOBAL (data.segments) so they see context.
               */}
              <ChatStream segments={data?.segments} />
            </div>
            <div className="lg:col-span-4 flex flex-col gap-6">
              <RiskRadar data={computedMetrics?.radar_chart_data || data?.radar_chart_data} />
              {(computedMetrics || data) && <ActionButtons data={computedMetrics || data} />}
            </div>
          </div>

        </div>

      </main>
    </div>
  );
}

export default App;
