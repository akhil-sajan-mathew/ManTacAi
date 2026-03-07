import React from 'react';
import { X, Phone, Globe, ShieldAlert } from 'lucide-react';
import { GlassCard } from './ui/GlassCard';

export function HelplineModal({ isOpen, onClose }) {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="relative w-full max-w-lg animate-in zoom-in-95 duration-200">
                <GlassCard className="p-6 border-rose-500/50 shadow-[0_0_30px_rgba(244,63,94,0.15)] overflow-hidden relative">
                    {/* Background glow */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-rose-500/10 rounded-full blur-3xl -tranneon-y-1/2 translate-x-1/3 pointer-events-none" />

                    {/* Header */}
                    <div className="flex justify-between items-start mb-6 relative z-10">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-rose-500/10 rounded-lg text-rose-500">
                                <ShieldAlert size={28} />
                            </div>
                            <div>
                                <h2 className="text-xl font-orbitron font-bold text-rose-400">Emergency Support</h2>
                                <p className="text-sm text-slate-400 font-sans">You are not alone. Help is available 24/7.</p>
                            </div>
                        </div>
                        <button
                            onClick={onClose}
                            className="p-1 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors"
                        >
                            <X size={24} />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="space-y-4 relative z-10">
                        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 flex items-start gap-4">
                            <Phone className="text-teal-400 mt-1" size={20} />
                            <div>
                                <h3 className="font-orbitron text-teal-300 font-semibold mb-1">National Domestic Violence Hotline (US)</h3>
                                <p className="text-3xl font-orbitron font-bold text-white mb-2 tracking-wider">800-799-7233</p>
                                <p className="text-sm text-slate-400">SMS: Text <span className="text-slate-200">"START"</span> to <span className="text-slate-200">88788</span></p>
                            </div>
                        </div>

                        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 flex items-start gap-4">
                            <Globe className="text-blue-400 mt-1" size={20} />
                            <div>
                                <h3 className="font-orbitron text-blue-300 font-semibold mb-1">International Resources</h3>
                                <a
                                    href="https://hotline.org/"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-white hover:text-blue-300 underline underline-offset-4 decoration-blue-500/50 transition-colors"
                                >
                                    Visit The Hotline Directory
                                </a>
                                <p className="text-sm text-slate-400 mt-2">Find localized support numbers and secure chat services.</p>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 text-xs text-slate-500 font-sans text-center relative z-10">
                        If you are in immediate physical danger, please contact your local emergency services (e.g., 911 in the US/Canada, 999 in the UK, 112 in Europe).
                    </div>
                </GlassCard>
            </div>
        </div>
    );
}
