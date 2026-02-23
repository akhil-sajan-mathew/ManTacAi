import React, { useEffect, useState } from 'react';
import { GlassCard } from './ui/GlassCard';
import { ShieldAlert, Activity, BrainCircuit } from 'lucide-react';
import { motion } from 'framer-motion';

function CountUp({ end, duration = 2 }) {
    const [count, setCount] = useState(0);

    useEffect(() => {
        let start = 0;
        const increment = end / (duration * 60);
        const timer = setInterval(() => {
            start += increment;
            if (start >= end) {
                setCount(end);
                clearInterval(timer);
            } else {
                setCount(start);
            }
        }, 1000 / 60);

        return () => clearInterval(timer);
    }, [end, duration]);

    return <>{count.toFixed(1)}</>;
}

export function HolographicHUD({ metrics }) {
    // Default metrics if null
    const {
        risk_score = 0,
        risk_level = "Safe",
        cycle_phase = "Scan Required",
        darvo_score = 0
    } = metrics || {};

    // Color Logic
    const getRiskColor = (score) => {
        if (score > 0.8) return "text-neon-red drop-shadow-[0_0_8px_rgba(244,63,94,0.8)]";
        if (score > 0.5) return "text-yellow-400 drop-shadow-[0_0_8px_rgba(250,204,21,0.6)]";
        return "text-neon-green drop-shadow-[0_0_8px_rgba(16,185,129,0.6)]";
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 w-full max-w-6xl">
            {/* Risk Score Card */}
            <GlassCard glow="red" delay={0.1} className="flex flex-col items-center justify-center border-t-2 border-t-neon-red/50">
                <ShieldAlert className="w-8 h-8 text-slate-400 mb-2" />
                <h3 className="text-sm font-orbitron text-slate-400 uppercase tracking-widest">Risk Level</h3>
                <div className={`text-4xl font-orbitron font-bold mt-2 ${getRiskColor(risk_score)}`}>
                    {risk_level.toUpperCase()}
                </div>
                <div className="text-xs text-slate-500 mt-1 font-mono">
                    CONFIDENCE: <CountUp end={risk_score * 100} />%
                </div>
            </GlassCard>

            {/* Cycle Phase Card */}
            <GlassCard glow="purple" delay={0.2} className="flex flex-col items-center justify-center border-t-2 border-t-neon-purple/50">
                <Activity className="w-8 h-8 text-slate-400 mb-2" />
                <h3 className="text-sm font-orbitron text-slate-400 uppercase tracking-widest">Active Phase</h3>
                <div className="text-3xl font-orbitron font-bold mt-2 text-white drop-shadow-[0_0_10px_rgba(255,255,255,0.4)]">
                    {cycle_phase}
                </div>
                <div className="text-xs text-neon-purple mt-1 font-mono uppercase tracking-wider">
                    {metrics?.primary_pattern || "SCANNING..."}
                </div>
            </GlassCard>

            {/* DARVO Index Card */}
            <GlassCard glow="cyan" delay={0.3} className="flex flex-col items-center justify-center border-t-2 border-t-neon-cyan/50">
                <BrainCircuit className="w-8 h-8 text-slate-400 mb-2" />
                <h3 className="text-sm font-orbitron text-slate-400 uppercase tracking-widest">DARVO Index</h3>
                <div className="text-4xl font-orbitron font-bold mt-2 text-neon-cyan drop-shadow-[0_0_8px_rgba(6,186,212,0.6)]">
                    <CountUp end={darvo_score} />
                </div>
                <div className="text-xs text-slate-500 mt-1 font-mono">
                    MANIPULATION DEPTH
                </div>
            </GlassCard>
        </div>
    );
}
