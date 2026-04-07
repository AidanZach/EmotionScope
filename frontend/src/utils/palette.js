/**
 * OKLCH emotion palette — tuned for luminous 3D rendering.
 *
 * Lightness carries emotional weight:
 *   Positive: L 0.70-0.88  (bright, lifted)
 *   Negative: L 0.35-0.55  (deep, heavy — but still visible against dark bg)
 *   Neutral:  L 0.58-0.68
 *
 * Chroma pushed to 0.20-0.28 for saturated, jewel-toned orbs.
 * On a dark background with bloom, higher chroma reads as luster, not garishness.
 */

import { oklchToRgb, oklchToHex } from './oklch';

export const PALETTE = {
  happy:        { L: 0.85, C: 0.22, H: 90 },
  loving:       { L: 0.78, C: 0.20, H: 25 },
  confident:    { L: 0.74, C: 0.18, H: 75 },
  proud:        { L: 0.74, C: 0.19, H: 70 },
  hopeful:      { L: 0.78, C: 0.16, H: 95 },
  calm:         { L: 0.68, C: 0.12, H: 220 },
  curious:      { L: 0.74, C: 0.16, H: 180 },
  reflective:   { L: 0.62, C: 0.06, H: 255 },
  surprised:    { L: 0.80, C: 0.18, H: 195 },
  enthusiastic: { L: 0.86, C: 0.24, H: 85 },
  nervous:      { L: 0.55, C: 0.14, H: 295 },
  frustrated:   { L: 0.52, C: 0.18, H: 35 },
  guilty:       { L: 0.42, C: 0.14, H: 315 },
  angry:        { L: 0.50, C: 0.24, H: 25 },
  sad:          { L: 0.45, C: 0.16, H: 270 },
  afraid:       { L: 0.40, C: 0.20, H: 300 },
  desperate:    { L: 0.38, C: 0.26, H: 15 },
  hostile:      { L: 0.35, C: 0.22, H: 18 },
  brooding:     { L: 0.40, C: 0.08, H: 265 },
  gloomy:       { L: 0.45, C: 0.06, H: 245 },
};

const NEUTRAL = { L: 0.58, C: 0.03, H: 240 };

/**
 * Convert an emotion name to a Three.js-compatible color string.
 */
export function emotionToColor(dominant) {
  const p = PALETTE[dominant] || NEUTRAL;
  return oklchToHex(p.L, p.C, p.H);
}

/**
 * Get the color from the backend's color_oklch object.
 */
export function oklchObjToHex(oklch) {
  if (!oklch) return '#555566';
  return oklchToHex(oklch.L, oklch.C, oklch.H);
}

/**
 * Map emotion state to R3F orb props.
 */
export function emotionToOrbProps(state) {
  if (!state || !state.dominant) {
    return {
      color: '#555566',
      emissive: '#222233',
      distort: 0.15,
      speed: 1.0,
      scale: 0.8,
      intensity: 0.2,
      arousal: 0,
      showParticles: false,
    };
  }

  const color = state.color_hex || emotionToColor(state.dominant);

  // Emissive should be BRIGHTER than the base — it's the glow source.
  // Boost lightness toward 1.0 and push chroma up for saturated inner glow.
  const baseL = state.color_oklch?.L || 0.5;
  const baseC = state.color_oklch?.C || 0.10;
  const baseH = state.color_oklch?.H || 240;
  const emL = Math.min(baseL * 1.25 + 0.1, 0.95);  // brighter, clamped
  const emC = Math.min(baseC * 1.3, 0.30);           // more saturated glow
  const [r, g, b] = oklchToRgb(emL, emC, baseH);
  const emissive = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;

  const arousal01 = ((state.arousal || 0) + 1) / 2;
  const intensity = Math.min(state.intensity || 0, 1);

  return {
    color,
    emissive,
    distort: 0.08 + (state.complexity || 0) * 0.45,
    speed: 0.5 + arousal01 * 3.5,
    scale: 0.7 + intensity * 0.6,
    intensity,
    arousal: state.arousal || 0,
    showParticles: (state.arousal || 0) > 0.4,
  };
}
