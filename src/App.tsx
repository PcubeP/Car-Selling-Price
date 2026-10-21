import { useEffect, useMemo, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { usePricePredictor } from '@/hooks/usePricePredictor'
import type { CarInput, CarModel, Schema } from '@/lib/schema'
import { formatINR, formatKm } from '@/lib/format'
import { cn } from '@/lib/cn'

export default function App() {
  const { schema, status, error, predict } = usePricePredictor()

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col px-6 py-14">
      <header className="mb-8">
        <div className="flex items-center gap-3">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-accent" />
          <span className="font-mono text-xs uppercase tracking-[0.3em] text-zinc-500">
            Resale Estimator · India
          </span>
        </div>
        <h1 className="mt-4 text-4xl font-semibold tracking-tight text-zinc-50 sm:text-5xl">
          What's your used car
          <br />
          <span className="text-accent">actually worth?</span>
        </h1>
        <p className="mt-4 max-w-xl text-zinc-400">
          A gradient-boosting model trained on India-market pricing patterns, exported to ONNX and
          running entirely in your browser. Adjust the details for a live estimate.
        </p>
      </header>

      {status === 'loading' && (
        <p className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3 font-mono text-xs text-zinc-400">
          Loading model…
        </p>
      )}
      {status === 'error' && (
        <p className="rounded-lg border border-red-900/60 bg-red-950/40 px-4 py-3 font-mono text-sm text-red-300">
          {error}
        </p>
      )}
      {status === 'ready' && schema && <Predictor schema={schema} predict={predict} />}

      <footer className="mt-12 font-mono text-[11px] leading-relaxed text-zinc-600">
        scikit-learn · ONNX Runtime Web · synthetic India-market dataset (demo). Estimates are
        illustrative, not financial advice.
      </footer>
    </div>
  )
}

function Predictor({
  schema,
  predict,
}: {
  schema: Schema
  predict: (input: CarInput) => Promise<number>
}) {
  const modelsByName = useMemo(
    () => new Map(schema.models.map((m) => [m.model, m])),
    [schema.models],
  )

  const first = schema.models[0]
  const [input, setInput] = useState<CarInput>(() => ({
    brand: first.brand,
    model: first.model,
    body_type: first.body_type,
    fuel: first.fuels[0],
    transmission: 'Manual',
    owner: 'First',
    year: schema.numeric.year.default,
    km_driven: schema.numeric.km_driven.default,
  }))
  const [price, setPrice] = useState<number | null>(null)

  const activeModel = modelsByName.get(input.model)!

  // Re-predict whenever any input changes.
  useEffect(() => {
    let cancelled = false
    predict(input)
      .then((p) => !cancelled && setPrice(p))
      .catch(() => !cancelled && setPrice(null))
    return () => {
      cancelled = true
    }
  }, [input, predict])

  function setModel(name: string) {
    const m = modelsByName.get(name) as CarModel
    setInput((prev) => ({
      ...prev,
      model: m.model,
      brand: m.brand,
      body_type: m.body_type,
      fuel: m.fuels.includes(prev.fuel) ? prev.fuel : m.fuels[0],
    }))
  }

  const grouped = useMemo(() => {
    const byBrand = new Map<string, CarModel[]>()
    for (const m of schema.models) {
      if (!byBrand.has(m.brand)) byBrand.set(m.brand, [])
      byBrand.get(m.brand)!.push(m)
    }
    return [...byBrand.entries()]
  }, [schema.models])

  const low = price != null ? price * (1 - schema.metrics.mape / 100) : null
  const high = price != null ? price * (1 + schema.metrics.mape / 100) : null

  return (
    <div className="space-y-6">
      <ResultCard price={price} low={low} high={high} car={`${activeModel.brand} ${activeModel.model}`} />

      <div className="grid gap-5 rounded-2xl border border-zinc-800 bg-zinc-900/40 p-6 sm:grid-cols-2">
        <Field label="Model">
          <select
            value={input.model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm"
          >
            {grouped.map(([brand, models]) => (
              <optgroup key={brand} label={brand}>
                {models.map((m) => (
                  <option key={m.model} value={m.model}>
                    {m.model}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </Field>

        <Field label="Fuel">
          <Segmented
            options={activeModel.fuels}
            value={input.fuel}
            onChange={(v) => setInput((p) => ({ ...p, fuel: v }))}
          />
        </Field>

        <Field label="Transmission">
          <Segmented
            options={schema.categorical.transmission}
            value={input.transmission}
            onChange={(v) => setInput((p) => ({ ...p, transmission: v }))}
          />
        </Field>

        <Field label="Owner">
          <Segmented
            options={schema.categorical.owner}
            value={input.owner}
            onChange={(v) => setInput((p) => ({ ...p, owner: v }))}
          />
        </Field>

        <Field label={`Year of purchase · ${input.year}`}>
          <input
            type="range"
            min={schema.numeric.year.min}
            max={schema.numeric.year.max}
            step={schema.numeric.year.step}
            value={input.year}
            onChange={(e) => setInput((p) => ({ ...p, year: Number(e.target.value) }))}
            className="w-full accent-accent"
          />
        </Field>

        <Field label={`Odometer · ${formatKm(input.km_driven)}`}>
          <input
            type="range"
            min={schema.numeric.km_driven.min}
            max={schema.numeric.km_driven.max}
            step={schema.numeric.km_driven.step}
            value={input.km_driven}
            onChange={(e) => setInput((p) => ({ ...p, km_driven: Number(e.target.value) }))}
            className="w-full accent-accent"
          />
        </Field>
      </div>

      <MetricsBar schema={schema} />
    </div>
  )
}

function ResultCard({
  price,
  low,
  high,
  car,
}: {
  price: number | null
  low: number | null
  high: number | null
  car: string
}) {
  return (
    <div className="rounded-2xl border border-accent/30 bg-gradient-to-b from-accent/10 to-transparent p-6">
      <p className="font-mono text-[11px] uppercase tracking-widest text-zinc-400">
        Estimated resale · {car}
      </p>
      <div className="mt-1 flex items-baseline gap-3">
        <AnimatePresence mode="popLayout">
          <motion.span
            key={price ?? 'na'}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className="text-4xl font-semibold text-zinc-50 sm:text-5xl"
          >
            {price != null ? formatINR(price) : '—'}
          </motion.span>
        </AnimatePresence>
      </div>
      {low != null && high != null && (
        <p className="mt-2 font-mono text-xs text-zinc-400">
          likely range {formatINR(low)} – {formatINR(high)}
        </p>
      )}
    </div>
  )
}

function MetricsBar({ schema }: { schema: Schema }) {
  const m = schema.metrics
  const items = [
    { label: 'R²', value: m.r2.toFixed(3) },
    { label: 'Mean error', value: `±${m.mape.toFixed(1)}%` },
    { label: 'Test cars', value: m.n_test.toLocaleString('en-IN') },
  ]
  return (
    <div className="flex flex-wrap gap-6 px-1 font-mono text-xs text-zinc-500">
      {items.map((it) => (
        <div key={it.label}>
          <span className="text-zinc-300">{it.value}</span> {it.label}
        </div>
      ))}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block space-y-2">
      <span className="font-mono text-[11px] uppercase tracking-wider text-zinc-400">{label}</span>
      {children}
    </label>
  )
}

function Segmented({
  options,
  value,
  onChange,
}: {
  options: string[]
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="inline-flex w-full rounded-lg border border-zinc-700 bg-zinc-950 p-1">
      {options.map((opt) => {
        const active = opt === value
        return (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={cn(
              'relative flex-1 rounded-md px-3 py-1.5 text-sm transition-colors',
              active ? 'text-zinc-900' : 'text-zinc-400 hover:text-zinc-200',
            )}
          >
            {active && (
              <motion.span
                layoutId={`seg-${options.join()}`}
                className="absolute inset-0 rounded-md bg-accent"
                transition={{ type: 'spring', stiffness: 400, damping: 32 }}
              />
            )}
            <span className="relative">{opt}</span>
          </button>
        )
      })}
    </div>
  )
}
