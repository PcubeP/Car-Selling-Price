import { useEffect, useRef, useState } from 'react'
import * as ort from 'onnxruntime-web'
import type { CarInput, Schema } from '@/lib/schema'

// Load the wasm backend from the CDN so we don't have to bundle/copy binaries.
// The wasm glue must match the installed onnxruntime-web version exactly, or it
// mismatches the JS ("e.getValue is not a function"). Derive the version from the
// package itself (onnxruntime-web reports its own version) so bumping the dep
// can't silently desync the CDN path.
const ORT_VERSION = ort.env.versions.web
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`

const CATEGORICAL = ['brand', 'model', 'body_type', 'fuel', 'transmission', 'owner'] as const
const NUMERIC = ['year', 'km_driven'] as const

type Status = 'loading' | 'ready' | 'error'

export function usePricePredictor() {
  const sessionRef = useRef<ort.InferenceSession | null>(null)
  // onnxruntime-web rejects concurrent run() on one session; serialize calls.
  const chainRef = useRef<Promise<unknown>>(Promise.resolve())
  const [schema, setSchema] = useState<Schema | null>(null)
  const [status, setStatus] = useState<Status>('loading')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const [schemaRes, session] = await Promise.all([
          fetch(`${import.meta.env.BASE_URL}model/schema.json`).then((r) => r.json() as Promise<Schema>),
          ort.InferenceSession.create(`${import.meta.env.BASE_URL}model/car_price.onnx`),
        ])
        if (cancelled) {
          // Effect was torn down (e.g. StrictMode double-mount) while the session
          // was still loading; release it so we don't orphan the native handle.
          void session.release()
          return
        }
        sessionRef.current = session
        setSchema(schemaRes)
        setStatus('ready')
      } catch (err) {
        if (cancelled) return
        setError(err instanceof Error ? err.message : String(err))
        setStatus('error')
      }
    }
    void load()
    return () => {
      cancelled = true
      // Release an already-loaded session; one still mid-load is released by the
      // cancelled guard above once create() resolves.
      const session = sessionRef.current
      sessionRef.current = null
      void session?.release()
    }
  }, [])

  async function runOnce(input: CarInput): Promise<number> {
    const session = sessionRef.current
    if (!session || !schema) throw new Error('Model not loaded')

    const feeds: Record<string, ort.Tensor> = {}
    for (const col of CATEGORICAL) {
      feeds[col] = new ort.Tensor('string', [String(input[col])], [1, 1])
    }
    for (const col of NUMERIC) {
      feeds[col] = new ort.Tensor('float32', Float32Array.from([Number(input[col])]), [1, 1])
    }
    const out = await session.run(feeds)
    const value = out[schema.outputName].data as Float32Array
    return value[0]
  }

  function predict(input: CarInput): Promise<number> {
    const next = chainRef.current.then(
      () => runOnce(input),
      () => runOnce(input),
    )
    chainRef.current = next.catch(() => undefined)
    return next
  }

  return { schema, status, error, predict }
}
