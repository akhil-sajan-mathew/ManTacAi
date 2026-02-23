import React from 'react';
import { GlassCard } from './ui/GlassCard';
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer
} from 'recharts';

export function RiskRadar({ data }) {
    // Default empty state if no data provided
    const chartData = data && data.length > 0 ? data : [
        { subject: 'Gaslighting', A: 0, fullMark: 100 },
        { subject: 'Guilt', A: 0, fullMark: 100 },
        { subject: 'Threats', A: 0, fullMark: 100 },
        { subject: 'Silence', A: 0, fullMark: 100 },
        { subject: 'Love Bomb', A: 0, fullMark: 100 },
        { subject: 'Deflection', A: 0, fullMark: 100 },
    ];

    return (
        <GlassCard className="h-[400px] flex flex-col w-full relative overflow-hidden">
            <h3 className="font-orbitron text-slate-400 uppercase tracking-widest text-sm mb-4 flex items-center gap-2 z-10">
                <span className="w-2 h-2 bg-neon-purple rounded-full" /> Tactic Fingerprint
            </h3>

            <div className="flex-1 w-full h-full min-h-[300px] -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="70%" data={chartData}>
                        <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                        <PolarAngleAxis
                            dataKey="subject"
                            tick={{ fill: '#94a3b8', fontSize: 10, fontFamily: 'Orbitron' }}
                        />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                        <Radar
                            name="Suspect"
                            dataKey="A"
                            stroke="#a855f7"
                            strokeWidth={2}
                            fill="#a855f7"
                            fillOpacity={0.5}
                        />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
            <div className="absolute inset-0 bg-gradient-radial from-neon-purple/5 to-transparent pointer-events-none" />
        </GlassCard>
    );
}
