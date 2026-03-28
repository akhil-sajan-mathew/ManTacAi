import { useMemo } from 'react';

/**
 * Custom hook that computes client-side aggregation metrics for a selected suspect.
 * Extracts the ~120-line computedMetrics logic from App.jsx into a reusable hook.
 *
 * @param {object|null} data - Raw analysis response from the backend
 * @param {string} selectedSuspect - Currently selected sender name or 'All'
 * @returns {object|null} Computed metrics object or null if no data
 */
export function useComputedMetrics(data, selectedSuspect) {
    return useMemo(() => {
        if (!data || !data.segments) return null;

        const allSegments = data.segments;
        const senders = [...new Set(allSegments.map(s => s.sender_name))];

        // STEP 1: CREATE THE "ACTIVE" SUBSET
        const activeSegments = selectedSuspect === 'All'
            ? allSegments
            : allSegments.filter(s => s.sender_name === selectedSuspect);

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
                cycle_phase: data.cycle_phase || "Normal",
                speaker_attribution: data.speaker_attribution || {}
            };
        }

        // STEP 2: CALCULATE METRICS *ONLY* ON THE SUBSET
        const riskScores = activeSegments.map(s => s.risk_score);
        const totalRisk = riskScores.reduce((sum, val) => sum + val, 0);
        const avgRisk = totalRisk / activeSegments.length;
        const maxRisk = Math.max(...riskScores);

        // STEP 3: APPLY "CRITICAL DOMINANCE" TO THE SUBSET
        let finalRiskInput = avgRisk;

        if (maxRisk > 0.85) {
            finalRiskInput = maxRisk; // Snap to Critical if THIS USER made a threat
        } else if (maxRisk > 0.60) {
            finalRiskInput = Math.max(avgRisk, 0.60); // Snap to Warning
        }

        // RISK FLOOR: In "All" view, the global risk must never go below the
        // server's undampened calculation (from aggregated tactic peaks).
        // This prevents dampened reactor scores from diluting critical peaks.
        if (selectedSuspect === 'All' && data.risk_score) {
            finalRiskInput = Math.max(finalRiskInput, data.risk_score);
        }

        // Determine Level
        let riskLevel = "Low";
        if (finalRiskInput > 0.8) riskLevel = "Critical";
        else if (finalRiskInput > 0.6) riskLevel = "High";
        else if (finalRiskInput > 0.3) riskLevel = "Medium";

        // STEP 4: Calc Radar Data (Subset Only)
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
        const primaryPattern = Object.entries(radarAgg).reduce((a, b) => {
            if (Math.abs(a[1] - b[1]) < 0.1) {
                const priority = { 'Threats': 3, 'Coercive Control': 3, 'Gaslighting': 2, 'Guilt': 2, 'Love Bomb': 1, 'Deflection': 1 };
                const pA = priority[a[0]] || 0;
                const pB = priority[b[0]] || 0;
                return pA >= pB ? a : b;
            }
            return a[1] > b[1] ? a : b;
        })[0] || "None";

        // STEP 5: Calc DARVO (Subset Only)
        const maxDarvo = activeSegments.reduce((max, s) => Math.max(max, s.darvo_score || 0), 0);

        // STEP 6: PHASE DETECTION OVERRIDE
        let localPhase = data.cycle_phase || "Normal";

        if (riskLevel === "Low") {
            localPhase = "Normal / Calm";
        } else if (riskLevel === "Medium") {
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
            cycle_phase: localPhase,
            speaker_attribution: data.speaker_attribution || {}
        };
    }, [data, selectedSuspect]);
}
