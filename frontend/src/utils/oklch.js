/**
 * OKLCH → sRGB conversion.
 * Used by the palette to convert the OKLCH emotion colors to CSS-usable values.
 */

export function oklchToRgb(L, C, H) {
  const hRad = (H * Math.PI) / 180;
  const a = C * Math.cos(hRad);
  const b = C * Math.sin(hRad);

  const l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const s_ = L - 0.0894841775 * a - 1.291485548 * b;

  let r = 4.0767416621 * l_ ** 3 - 3.3077115913 * m_ ** 3 + 0.2309699292 * s_ ** 3;
  let g = -1.2684380046 * l_ ** 3 + 2.6097574011 * m_ ** 3 - 0.3413193965 * s_ ** 3;
  let bl = -0.0041960863 * l_ ** 3 - 0.7034186147 * m_ ** 3 + 1.707614701 * s_ ** 3;

  const gamma = (v) => (v > 0.0031308 ? 1.055 * v ** (1 / 2.4) - 0.055 : 12.92 * v);

  return [
    Math.max(0, Math.min(1, gamma(r))),
    Math.max(0, Math.min(1, gamma(g))),
    Math.max(0, Math.min(1, gamma(bl))),
  ];
}

export function oklchToHex(L, C, H) {
  const [r, g, b] = oklchToRgb(L, C, H);
  const toHex = (v) =>
    Math.round(v * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

export function oklchToCssRgb(L, C, H) {
  const [r, g, b] = oklchToRgb(L, C, H);
  return `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
}

/**
 * Circular hue interpolation via vector averaging (VISUALIZATION_SPEC.md §9).
 */
export function blendHues(entries) {
  // entries: [{H, weight}, ...]
  let hx = 0, hy = 0, tw = 0;
  for (const { H, weight } of entries) {
    hx += Math.cos((H * Math.PI) / 180) * weight;
    hy += Math.sin((H * Math.PI) / 180) * weight;
    tw += weight;
  }
  if (tw === 0) return 240;
  return ((Math.atan2(hy / tw, hx / tw) * 180) / Math.PI + 360) % 360;
}
