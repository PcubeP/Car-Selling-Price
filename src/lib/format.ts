/** Format a rupee amount the Indian way: ₹X.XX Lakh / ₹X.XX Crore. */
export function formatINR(value: number): string {
  if (!isFinite(value)) return '—'
  if (value >= 1e7) return `₹${(value / 1e7).toFixed(2)} Cr`
  if (value >= 1e5) return `₹${(value / 1e5).toFixed(2)} Lakh`
  return `₹${Math.round(value).toLocaleString('en-IN')}`
}

/** Compact odometer label, e.g. 45,000 km. */
export function formatKm(value: number): string {
  return `${Math.round(value).toLocaleString('en-IN')} km`
}
