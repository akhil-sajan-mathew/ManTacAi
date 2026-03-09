import { describe, it, expect } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useComputedMetrics } from '../hooks/useComputedMetrics';

const makeSeg = (overrides = {}) => ({
    msg: 'Test message',
    ts: 1700000000,
    sender: 'suspect',
    sender_name: 'Alex',
    risk_score: 0.5,
    label: 'gaslighting',
    timestamp_str: '10:30',
    tactic_scores: { gaslighting: 0.5, guilt_tripping: 0.2 },
    darvo_score: 0.3,
    ...overrides,
});

describe('useComputedMetrics', () => {
    it('returns null when data is null', () => {
        const { result } = renderHook(() => useComputedMetrics(null, 'All'));
        expect(result.current).toBeNull();
    });

    it('returns null when data has no segments', () => {
        const { result } = renderHook(() => useComputedMetrics({}, 'All'));
        expect(result.current).toBeNull();
    });

    it('computes metrics for all segments', () => {
        const data = {
            segments: [makeSeg(), makeSeg({ sender_name: 'You', sender: 'victim', risk_score: 0.1 })],
            cycle_phase: 'NORMAL',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'All'));
        expect(result.current).not.toBeNull();
        expect(result.current.senders).toContain('All');
        expect(result.current.senders).toContain('Alex');
        expect(result.current.senders).toContain('You');
        expect(result.current.segments.length).toBe(2);
    });

    it('filters segments by selected suspect', () => {
        const data = {
            segments: [
                makeSeg({ sender_name: 'Alex', risk_score: 0.8 }),
                makeSeg({ sender_name: 'You', sender: 'victim', risk_score: 0.1 }),
            ],
            cycle_phase: 'TENSION',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'Alex'));
        expect(result.current.segments.length).toBe(1);
        expect(result.current.segments[0].sender_name).toBe('Alex');
    });

    it('returns safe state for empty filtered segments', () => {
        const data = {
            segments: [makeSeg({ sender_name: 'Alex' })],
            cycle_phase: 'NORMAL',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'NonexistentPerson'));
        expect(result.current.risk_score).toBe(0);
        expect(result.current.risk_level).toBe('Safe');
    });

    it('snaps to Critical when max risk > 0.85', () => {
        const data = {
            segments: [makeSeg({ risk_score: 0.9 }), makeSeg({ risk_score: 0.1 })],
            cycle_phase: 'EXPLOSION',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'All'));
        expect(result.current.risk_level).toBe('Critical');
        expect(result.current.risk_score).toBe(0.9);
    });

    it('generates radar chart data', () => {
        const data = {
            segments: [makeSeg({ tactic_scores: { gaslighting: 0.8, guilt_tripping: 0.4 } })],
            cycle_phase: 'NORMAL',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'All'));
        const gaslight = result.current.radar_chart_data.find(d => d.subject === 'Gaslighting');
        expect(gaslight).toBeDefined();
        expect(gaslight.A).toBe(80);
    });

    it('overrides phase to Normal/Calm for low risk', () => {
        const data = {
            segments: [makeSeg({ risk_score: 0.1, tactic_scores: {} })],
            cycle_phase: 'EXPLOSION',
        };
        const { result } = renderHook(() => useComputedMetrics(data, 'All'));
        expect(result.current.cycle_phase).toBe('Normal / Calm');
    });
});
