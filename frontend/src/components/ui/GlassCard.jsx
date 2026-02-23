import React from 'react';
import { motion } from 'framer-motion';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function GlassCard({ children, className, glow = "purple", delay = 0 }) {
    const glowColors = {
        purple: "hover:border-neon-purple/50 hover:shadow-[0_0_20px_rgba(168,85,247,0.2)]",
        red: "hover:border-neon-red/50 hover:shadow-[0_0_20px_rgba(244,63,94,0.2)]",
        cyan: "hover:border-neon-cyan/50 hover:shadow-[0_0_20px_rgba(6,182,212,0.2)]",
        none: ""
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: delay }}
            className={twMerge(
                "glass-panel p-6 transition-all duration-300 relative overflow-hidden group",
                glowColors[glow],
                className
            )}
        >
            {/* Scanline Effect Overlay (Optional) */}
            <div className="absolute inset-0 bg-gradient-to-b from-white/5 to-transparent opacity-0 group-hover:opacity-10 pointer-events-none transition-opacity duration-500" />

            {/* Content Buffer to ensure interactivity */}
            <div className="relative z-10">
                {children}
            </div>
        </motion.div>
    );
}
