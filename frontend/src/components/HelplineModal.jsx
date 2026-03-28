import React, { useEffect, useRef } from 'react';
import { X, Phone, Globe, ShieldAlert } from 'lucide-react';
import { GlassCard } from './ui/GlassCard';

export function HelplineModal({ isOpen, onClose }) {
    const closeRef = useRef(null);

    // Focus trap: focus the close button when modal opens
    useEffect(() => {
        if (isOpen && closeRef.current) {
            closeRef.current.focus();
        }
    }, [isOpen]);

    // Close on Escape key
    useEffect(() => {
        if (!isOpen) return;
        const handleKey = (e) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200"
            role="dialog"
            aria-modal="true"
            aria-labelledby="helpline-modal-title"
            onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
        >
            <div className="relative w-full max-w-lg animate-in zoom-in-95 duration-200">
                <GlassCard className="p-6 border-rose-500/50 shadow-[0_0_30px_rgba(244,63,94,0.15)] overflow-hidden relative">
                    {/* Background glow */}
                    <div className="absolute top-0 right-0 w-64 h-64 bg-rose-500/10 rounded-full blur-3xl -tranneon-y-1/2 translate-x-1/3 pointer-events-none" />

                    {/* Header */}
                    <div className="flex justify-between items-start mb-6 relative z-10">
                        <div className="flex items-center gap-3">
                            <div className="p-2 bg-rose-500/10 rounded-lg text-rose-500">
                                <ShieldAlert size={28} aria-hidden="true" />
                            </div>
                            <div>
                                <h2 id="helpline-modal-title" className="text-xl font-orbitron font-bold text-rose-400">Emergency Support</h2>
                                <p className="text-sm text-slate-400 font-sans">You are not alone. Help is available 24/7.</p>
                            </div>
                        </div>
                        <button
                            ref={closeRef}
                            onClick={onClose}
                            aria-label="Close emergency support dialog"
                            className="p-1 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors"
                        >
                            <X size={24} aria-hidden="true" />
                        </button>
                    </div>

                    {/* Content */}
                    <div className="space-y-4 relative z-10">
                        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 flex items-start gap-4">
                            <Phone className="text-teal-400 mt-1" size={20} aria-hidden="true" />
                            <div>
                                <h3 className="font-orbitron text-teal-300 font-semibold mb-1">National Women's Helpline (India)</h3>
                                <p className="text-3xl font-orbitron font-bold text-white mb-2 tracking-wider">
                                    <a href="tel:181" aria-label="Call National Women's Helpline at 181">181</a>
                                </p>
                                <p className="text-sm text-slate-400">Police Emergency: <span className="text-slate-200">112</span> | Women Police: <span className="text-slate-200">1091</span></p>
                            </div>
                        </div>

                        <div className="p-4 rounded-xl bg-slate-900/50 border border-slate-700/50 flex items-start gap-4">
                            <Globe className="text-blue-400 mt-1" size={20} aria-hidden="true" />
                            <div>
                                <h3 className="font-orbitron text-blue-300 font-semibold mb-1">Kerala State Resources</h3>
                                <div className="space-y-2">
                                    <p className="text-sm text-slate-400">Mitra 181 (Kerala): <a href="tel:181" className="text-white hover:text-blue-300 transition-colors font-medium">181</a></p>
                                    <p className="text-sm text-slate-400">Kerala Women's Commission: <a href="tel:04712300509" className="text-white hover:text-blue-300 transition-colors font-medium">0471-2300509</a></p>
                                    <p className="text-sm text-slate-400">Aparajitha is cyber support for women: <a href="mailto:aparajitha.pol@kerala.gov.in" className="text-white hover:text-blue-300 transition-colors font-medium">aparajitha.pol@kerala.gov.in</a></p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 text-xs text-slate-500 font-sans text-center relative z-10">
                        If you are in immediate physical danger, please contact local emergency services immediately (Dial 112).
                    </div>
                </GlassCard>
            </div>
        </div>
    );
}

