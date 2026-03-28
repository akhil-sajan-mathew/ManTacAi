import React, { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { User, ShieldAlert, Bot } from 'lucide-react';

export function ChatStream({ segments }) {
    const scrollRef = useRef(null);

    // Auto-scroll to bottom
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [segments]);

    if (!segments || segments.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center p-10 h-96 text-slate-500 border border-dashed border-glass-border rounded-xl">
                <Bot className="w-12 h-12 mb-4 opacity-50" />
                <p className="font-orbitron tracking-widest text-sm">AWAITING DATA INPUT...</p>
            </div>
        );
    }

    return (
        <div className="relative glass-panel h-[600px] flex flex-col overflow-hidden">
            {/* Header */}
            <div className="p-4 border-b border-glass-border flex justify-between items-center bg-black/20">
                <h3 className="font-orbitron font-bold text-neon-cyan tracking-wider flex items-center gap-2">
                    <ActivityDot /> LIVE ANALYSIS STREAM
                </h3>
                <span className="text-xs font-mono text-slate-400">
                    EVENTS: {segments.length}
                </span>
            </div>

            {/* Stream Area */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-thin scrollbar-thumb-neon-purple/20"
            >
                <AnimatePresence initial={false}>
                    {segments.map((seg, idx) => (
                        <ChatBubble key={`${seg.ts}-${seg.sender_name}-${idx}`} segment={seg} index={idx} />
                    ))}
                </AnimatePresence>
            </div>
        </div>
    );
}

function ChatBubble({ segment, index }) {
    // Robust check for "Me"
    const isMe = segment.sender === 'victim';
    const isHighRisk = segment.risk_score > 0.7;
    const role = segment.role || 'neutral';
    const isInitiator = role === 'initiator';
    const isReactor = role === 'reactor';

    const senderName = segment.sender_name || (isMe ? 'You' : 'Unknown');
    const initial = senderName.charAt(0).toUpperCase();

    // Left border color based on role
    const roleBorderClass = isInitiator && segment.risk_score > 0.3
        ? 'border-l-2 border-l-red-500'
        : isReactor
            ? 'border-l-2 border-l-amber-500/60'
            : '';

    return (
        <motion.div
            initial={{ opacity: 0, x: isMe ? 20 : -20, scale: 0.95 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            transition={{ duration: 0.3, delay: Math.min(index * 0.05, 1.5) }}
            className={`flex w-full group ${isMe ? 'justify-end' : 'justify-start'}`}
        >
            {/* Avatar for Non-Me Users */}
            {!isMe && (
                <div className={`
                    w-8 h-8 rounded-full flex items-center justify-center mr-3 mt-1 shadow-lg border
                    ${segment.sender === 'suspect'
                        ? 'bg-red-900/50 border-neon-red text-neon-red'
                        : 'bg-slate-800 border-slate-600 text-slate-400'}
                `}>
                    <span className="font-orbitron font-bold text-xs">{initial}</span>
                </div>
            )}

            <div className={`
        relative max-w-[80%] rounded-2xl p-4 border backdrop-blur-md shadow-lg ${roleBorderClass}
        ${isMe
                    ? 'bg-slate-900/60 border-neon-cyan/30 text-slate-200 rounded-tr-sm'
                    : isHighRisk
                        ? 'bg-red-950/40 border-neon-red text-red-100 rounded-tl-sm shadow-[0_0_15px_rgba(244,63,94,0.15)]'
                        : 'bg-slate-900/60 border-slate-700/50 text-slate-300 rounded-tl-sm'
                }
      `}>
                {/* Risk Badge */}
                {isHighRisk && (
                    <div className="absolute -top-3 -right-2 bg-neon-red text-black text-[10px] font-bold px-2 py-0.5 rounded-full border border-red-500 shadow-md flex items-center gap-1 font-orbitron">
                        <ShieldAlert className="w-3 h-3" /> ALERT
                    </div>
                )}

                {/* Sender Label */}
                <div className="flex items-center gap-2 mb-1 opacity-70 text-xs font-mono uppercase tracking-wider">
                    <span className={isMe ? "text-neon-cyan" : "text-white"}>
                        {senderName}
                    </span>
                    <span className="opacity-50">• {segment.timestamp_str}</span>
                </div>

                {/* Message Content */}
                <p className="whitespace-pre-wrap leading-relaxed">
                    {segment.msg}
                </p>

                {/* Detection Tag Footer + Role Badge */}
                {!isMe && segment.label && segment.risk_score > 0.4 && (
                    <div className={`mt-3 pt-2 border-t ${isHighRisk ? 'border-red-500/20' : 'border-slate-700/30'} flex justify-between items-center`}>
                        <div className="flex items-center gap-2">
                            <span className={`text-[10px] uppercase font-bold ${isHighRisk ? 'text-neon-red' : 'text-slate-400'}`}>
                                Detected: {segment.label.replace(/_/g, " ")}
                            </span>
                            {isInitiator && (
                                <span className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
                                    ⚡ initiated
                                </span>
                            )}
                            {isReactor && (
                                <span className="text-[9px] font-bold uppercase px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-400/80 border border-amber-500/25">
                                    🛡️ reactive
                                </span>
                            )}
                        </div>
                        <div className="h-1 w-16 bg-black/50 rounded-full overflow-hidden">
                            <div
                                className={`h-full ${isHighRisk ? 'bg-neon-red' : 'bg-neon-purple'}`}
                                style={{ width: `${segment.risk_score * 100}%` }}
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Avatar for Me (Right side) */}
            {isMe && (
                <div className="w-8 h-8 rounded-full bg-neon-cyan/20 border border-neon-cyan text-neon-cyan flex items-center justify-center ml-3 mt-1 shadow-[0_0_10px_rgba(6,186,212,0.3)]">
                    <span className="font-orbitron font-bold text-xs">{initial}</span>
                </div>
            )}
        </motion.div>
    );
}

function ActivityDot() {
    return (
        <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-neon-cyan opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-neon-cyan"></span>
        </span>
    );
}
