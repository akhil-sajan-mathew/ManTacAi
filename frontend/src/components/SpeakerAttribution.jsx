import React from 'react';
import { GlassCard } from './ui/GlassCard';
import { Users, Zap, Shield } from 'lucide-react';

/**
 * Displays per-speaker power dynamics attribution.
 * Shows initiation ratio, tactic diversity, and avg risk for each speaker.
 */
export function SpeakerAttribution({ attribution }) {
    if (!attribution || Object.keys(attribution).length === 0) return null;

    const speakers = Object.entries(attribution)
        .filter(([, data]) => data.message_count > 0)
        .sort((a, b) => (b[1].initiation_ratio || 0) - (a[1].initiation_ratio || 0));

    if (speakers.length === 0) return null;

    return (
        <GlassCard className="p-4">
            <h3 className="font-orbitron text-xs tracking-widest uppercase text-slate-400 mb-3 flex items-center gap-2">
                <Users className="w-3.5 h-3.5 text-neon-purple" />
                Power Dynamics
            </h3>

            <div className="space-y-3">
                {speakers.map(([name, data]) => {
                    const initRatio = data.initiation_ratio || 0;
                    const reactRatio = 1 - initRatio;
                    const totalFlagged = (data.initiation_count || 0) + (data.reaction_count || 0);

                    // Determine dominance label
                    let dominanceLabel = 'Neutral';
                    let dominanceColor = 'text-slate-400';
                    if (initRatio > 0.6 && totalFlagged > 0) {
                        dominanceLabel = 'Aggressor';
                        dominanceColor = 'text-red-400';
                    } else if (initRatio < 0.4 && totalFlagged > 0) {
                        dominanceLabel = 'Defensive';
                        dominanceColor = 'text-amber-400';
                    }

                    return (
                        <div key={name} className="border border-slate-700/40 rounded-lg p-3 bg-black/20">
                            {/* Speaker Name + Dominance Label */}
                            <div className="flex justify-between items-center mb-2">
                                <span className="font-mono text-sm text-white font-semibold">
                                    {name}
                                </span>
                                <span className={`text-[10px] font-bold uppercase tracking-wider ${dominanceColor}`}>
                                    {dominanceLabel}
                                </span>
                            </div>

                            {/* Initiation vs Reaction Bar */}
                            {totalFlagged > 0 && (
                                <div className="mb-2">
                                    <div className="flex justify-between text-[10px] font-mono text-slate-500 mb-1">
                                        <span className="flex items-center gap-1">
                                            <Zap className="w-2.5 h-2.5 text-red-400" />
                                            Initiated {Math.round(initRatio * 100)}%
                                        </span>
                                        <span className="flex items-center gap-1">
                                            Reactive {Math.round(reactRatio * 100)}%
                                            <Shield className="w-2.5 h-2.5 text-amber-400" />
                                        </span>
                                    </div>
                                    <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden flex">
                                        <div
                                            className="h-full bg-red-500/80 transition-all duration-500"
                                            style={{ width: `${initRatio * 100}%` }}
                                        />
                                        <div
                                            className="h-full bg-amber-500/50 transition-all duration-500"
                                            style={{ width: `${reactRatio * 100}%` }}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Stats Row */}
                            <div className="flex gap-3 text-[10px] font-mono text-slate-500">
                                <span>{data.message_count || 0} msgs</span>
                                <span>•</span>
                                <span>{data.tactic_diversity || 0} tactics</span>
                                <span>•</span>
                                <span>avg risk {Math.round((data.avg_risk || 0) * 100)}%</span>
                            </div>
                        </div>
                    );
                })}
            </div>
        </GlassCard>
    );
}
