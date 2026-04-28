'use client'

import { Suspense, useEffect, useMemo, useState } from 'react'
import Image from 'next/image'
import { useRouter, useSearchParams } from 'next/navigation'
import GlassSurface from '@/components/GlassSurface'
import {
  AreaUnit,
  FieldMetricsResult,
  FieldMetricsState,
  calculateFieldMetrics,
  fieldMetricsStorageKey,
} from '@/lib/api'
import { CONVOCROP_LOGO_SRC } from '@/lib/assets'

type RowKey = `${number}-${number}`

function parseRowKey(k: string): { q: number; s: number } | null {
  const m = /^(\d+)-(\d+)$/.exec(k)
  if (!m) return null
  return { q: Number(m[1]), s: Number(m[2]) }
}

function NarrativeBlock({ text }: { text: string }) {
  const segments = text.split(/(\*\*[^*]+\*\*)/g)
  return (
    <div className="text-sm font-medium text-[#252525] leading-relaxed whitespace-pre-wrap break-words">
      {segments.map((seg, i) => {
        if (seg.startsWith('**') && seg.endsWith('**') && seg.length > 4) {
          return (
            <strong key={i} className="font-semibold text-[#252525]">
              {seg.slice(2, -2)}
            </strong>
          )
        }
        return <span key={i}>{seg}</span>
      })}
    </div>
  )
}

function FieldSeverityContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const sessionId = searchParams.get('sessionId')

  const [totalFieldArea, setTotalFieldArea] = useState('')
  const [areaUnit, setAreaUnit] = useState<AreaUnit>('m2')
  const [samplesPerQuadrant, setSamplesPerQuadrant] = useState(3)
  const [counts, setCounts] = useState<Record<RowKey, { total: string; infected: string }>>({})

  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<FieldMetricsResult | null>(null)
  const [metricsState, setMetricsState] = useState<FieldMetricsState | null>(null)

  const rowKeys = useMemo(() => {
    const keys: RowKey[] = []
    for (let q = 1; q <= 4; q++) {
      for (let s = 1; s <= samplesPerQuadrant; s++) {
        keys.push(`${q}-${s}` as RowKey)
      }
    }
    return keys
  }, [samplesPerQuadrant])

  useEffect(() => {
    setCounts((prev) => {
      const next = { ...prev } as Record<RowKey, { total: string; infected: string }>
      for (const k of rowKeys) {
        if (!next[k]) next[k] = { total: '', infected: '' }
      }
      for (const k of Object.keys(next)) {
        if (!rowKeys.includes(k as RowKey)) delete next[k as RowKey]
      }
      return next
    })
  }, [rowKeys])

  function updateCell(key: RowKey, field: 'total' | 'infected', value: string) {
    if (!/^\d*$/.test(value)) return
    setCounts((prev) => ({
      ...prev,
      [key]: { ...prev[key], [field]: value },
    }))
  }

  async function handleCalculate() {
    setError(null)
    const area = parseFloat(totalFieldArea)
    if (!Number.isFinite(area) || area <= 0) {
      setError('Enter a valid total field area greater than zero.')
      return
    }

    const samples: {
      quadrant_id: number
      total_plants: number
      infected_plants: number
    }[] = []

    for (const key of rowKeys) {
      const row = counts[key]
      if (!row) continue
      const total = parseInt(row.total, 10)
      const infected = parseInt(row.infected, 10)
      if (!Number.isFinite(total) || total <= 0) {
        setError('Each sample needs a positive total plant count.')
        return
      }
      if (!Number.isFinite(infected) || infected < 0) {
        setError('Infected counts must be zero or a positive whole number.')
        return
      }
      if (infected > total) {
        setError('Infected plants cannot exceed total plants in any sample.')
        return
      }
      const parsed = parseRowKey(key)
      if (!parsed) continue
      samples.push({
        quadrant_id: parsed.q,
        total_plants: total,
        infected_plants: infected,
      })
    }

    if (samples.length === 0) {
      setError('Add at least one complete sample row.')
      return
    }

    setLoading(true)
    try {
      const res = await calculateFieldMetrics({
        total_field_area: area,
        area_unit: areaUnit,
        samples,
      })
      setResult(res.field_metrics)
      setMetricsState(res.field_metrics_state)
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? (e as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null
      setError(typeof msg === 'string' ? msg : 'Could not calculate metrics. Try again.')
      setResult(null)
      setMetricsState(null)
    } finally {
      setLoading(false)
    }
  }

  function continueInChat() {
    if (!metricsState) return
    if (sessionId) {
      try {
        sessionStorage.setItem(fieldMetricsStorageKey(sessionId), JSON.stringify(metricsState))
      } catch {
        /* ignore quota */
      }
      router.push(`/chat?sessionId=${encodeURIComponent(sessionId)}&fieldMetrics=1`)
      return
    }
    router.push('/chat')
  }

  return (
    <div className="min-h-screen w-full relative flex flex-col bg-[#ebebeb]">
      <div className="fixed inset-0 z-0 bg-[#ebebeb]" />

      <header className="relative z-10 flex items-center justify-between gap-3 px-4 py-3 border-b border-[#252525]/10 shrink-0"
        style={{ background: 'rgba(235,235,235,0.85)', backdropFilter: 'blur(12px)' }}
      >
        <button
          type="button"
          onClick={() => router.push('/')}
          className="flex items-center gap-2 text-sm text-[#252525]/70 hover:text-[#252525]"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Home
        </button>
        <div className="relative h-8 w-28">
          <Image src={CONVOCROP_LOGO_SRC} alt="Convo Crop" fill className="object-contain" priority />
        </div>
        <div className="w-16" aria-hidden />
      </header>

      <main className="relative z-10 flex-1 w-full max-w-3xl mx-auto px-4 py-6 md:py-10 pb-24">
        <h1 className="text-2xl md:text-3xl font-bold text-[#252525] mb-2">Field disease severity</h1>
        <p className="text-sm text-[#252525]/65 mb-6">
          Enter your quadrat counts below. Results sync with your chat when you continue.
        </p>

        <section
          className="rounded-2xl border border-white/50 bg-white/25 p-4 md:p-6 mb-6 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.45),0_8px_32px_rgba(37,37,37,0.06)]"
          style={{ backdropFilter: 'blur(20px)' }}
        >
          <h2 className="text-sm font-semibold text-[#252525] uppercase tracking-wide mb-3">
            How to take samples
          </h2>
          <ol className="list-decimal pl-5 space-y-2 text-sm text-[#252525]/85 leading-relaxed">
            <li>
              Mentally divide the field into <strong className="font-semibold">four equal parts</strong> (quadrants:
              e.g. north-west, north-east, south-west, south-east).
            </li>
            <li>
              In each quadrant, place a <strong className="font-semibold">fixed-size plot</strong> (quadrat) or use a
              hoop with a known radius so each sample covers the same ground area.
            </li>
            <li>
              For each planned sample spot, count <strong className="font-semibold">all plants</strong> inside the
              quadrat, then count how many show clear disease symptoms (infected).
            </li>
            <li>
              Pick sample locations <strong className="font-semibold">at random</strong> within each quadrant, not only
              the worst patches, so the average reflects the whole field.
            </li>
            <li>
              Use the same number of samples in every quadrant. Enter your total field size in the unit you normally
              use (kanal, marla, square meters, acres, or hectares).
            </li>
          </ol>
        </section>

        <section className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-semibold text-[#252525]/70 mb-1.5">Total field area</label>
              <input
                type="text"
                inputMode="decimal"
                value={totalFieldArea}
                onChange={(e) => {
                  const v = e.target.value
                  if (v === '' || /^\d*\.?\d*$/.test(v)) setTotalFieldArea(v)
                }}
                className="w-full rounded-xl border border-white/55 bg-white/35 px-3 py-2.5 text-sm text-[#252525] shadow-[inset_0_1px_2px_rgba(255,255,255,0.5)] focus:outline-none focus:ring-2 focus:ring-[#252525]/12"
                style={{ backdropFilter: 'blur(14px)' }}
                placeholder="e.g. 2.5"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-[#252525]/70 mb-1.5">Unit</label>
              <select
                value={areaUnit}
                onChange={(e) => setAreaUnit(e.target.value as AreaUnit)}
                className="w-full rounded-xl border border-white/55 bg-white/35 px-3 py-2.5 text-sm text-[#252525] shadow-[inset_0_1px_2px_rgba(255,255,255,0.5)] focus:outline-none focus:ring-2 focus:ring-[#252525]/12"
                style={{ backdropFilter: 'blur(14px)' }}
              >
                <option value="m2">Square meters (m²)</option>
                <option value="kanal">Kanal</option>
                <option value="marla">Marla</option>
                <option value="acre">Acre</option>
                <option value="ha">Hectare (ha)</option>
              </select>
            </div>
          </div>

          <div className="max-w-xs">
            <label className="block text-xs font-semibold text-[#252525]/70 mb-1.5">
              Number of samples per quadrant
            </label>
            <input
              type="number"
              min={1}
              max={20}
              value={samplesPerQuadrant}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10)
                if (Number.isFinite(n) && n >= 1 && n <= 20) setSamplesPerQuadrant(n)
              }}
              className="w-full rounded-xl border border-white/55 bg-white/35 px-3 py-2.5 text-sm text-[#252525] shadow-[inset_0_1px_2px_rgba(255,255,255,0.5)] focus:outline-none focus:ring-2 focus:ring-[#252525]/12"
              style={{ backdropFilter: 'blur(14px)' }}
            />
            <p className="text-xs text-[#252525]/50 mt-1">
              Four quadrants × this value = total samples ({4 * samplesPerQuadrant}).
            </p>
          </div>

          <div
            className="space-y-4 rounded-2xl border border-white/50 bg-white/20 p-4 md:p-5 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.4),0_8px_32px_rgba(37,37,37,0.05)]"
            style={{ backdropFilter: 'blur(20px)' }}
          >
            <p className="text-sm font-medium text-[#252525] mb-3">Sample counts</p>
            {[1, 2, 3, 4].map((q) => (
              <div
                key={q}
                className="rounded-2xl border border-white/45 bg-white/25 p-4 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.35)]"
                style={{ backdropFilter: 'blur(12px)' }}
              >
                <p className="text-xs font-semibold text-[#252525]/80 mb-3">Quadrant {q}</p>
                <div className="space-y-3">
                  {Array.from({ length: samplesPerQuadrant }, (_, i) => i + 1).map((s) => {
                    const key = `${q}-${s}` as RowKey
                    const row = counts[key] ?? { total: '', infected: '' }
                    return (
                      <div
                        key={key}
                        className="grid grid-cols-1 gap-3 sm:grid-cols-3 sm:items-end"
                      >
                        <p className="text-xs text-[#252525]/55 sm:col-span-1 pt-2">Sample {s}</p>
                        <div>
                          <label className="block text-[10px] uppercase tracking-wide text-[#252525]/55 mb-1">
                            Total plants
                          </label>
                          <input
                            type="text"
                            inputMode="numeric"
                            value={row.total}
                            onChange={(e) => updateCell(key, 'total', e.target.value)}
                            className="w-full rounded-lg border border-white/50 bg-white/30 px-2.5 py-2 text-sm text-[#252525] shadow-[inset_0_1px_2px_rgba(255,255,255,0.45)]"
                            style={{ backdropFilter: 'blur(10px)' }}
                          />
                        </div>
                        <div>
                          <label className="block text-[10px] uppercase tracking-wide text-[#252525]/55 mb-1">
                            Infected plants
                          </label>
                          <input
                            type="text"
                            inputMode="numeric"
                            value={row.infected}
                            onChange={(e) => updateCell(key, 'infected', e.target.value)}
                            className="w-full rounded-lg border border-white/50 bg-white/30 px-2.5 py-2 text-sm text-[#252525] shadow-[inset_0_1px_2px_rgba(255,255,255,0.45)]"
                            style={{ backdropFilter: 'blur(10px)' }}
                          />
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>

          {error && (
            <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">{error}</div>
          )}

          <GlassSurface
            width="100%"
            height={48}
            borderRadius={14}
            onClick={loading ? undefined : handleCalculate}
            className={loading ? 'opacity-60 pointer-events-none' : 'cursor-pointer'}
          >
            <span className="text-[#252525] font-semibold text-sm">
              {loading ? 'Calculating...' : 'Calculate severity'}
            </span>
          </GlassSurface>

          {result && (
            <section
              className="rounded-2xl border border-[#00a71b]/25 bg-white/60 p-4 md:p-6 mt-6"
              style={{ backdropFilter: 'blur(8px)' }}
            >
              <h2 className="text-lg font-bold text-[#252525] mb-4">Results</h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4 text-sm">
                <div className="rounded-xl bg-[#00a71b]/8 px-3 py-2 border border-[#00a71b]/15">
                  <p className="text-[10px] uppercase text-[#252525]/55">Average incidence</p>
                  <p className="text-lg font-semibold text-[#252525]">{result.average_incidence_pct.toFixed(1)}%</p>
                </div>
                <div className="rounded-xl bg-[#00a71b]/8 px-3 py-2 border border-[#00a71b]/15">
                  <p className="text-[10px] uppercase text-[#252525]/55">Estimated affected area</p>
                  <p className="text-lg font-semibold text-[#252525]">
                    {result.estimated_infected_area.toFixed(2)} {result.estimated_infected_area_unit}
                  </p>
                </div>
              </div>
              <NarrativeBlock text={result.narrative} />

              <div className="mt-6">
                <GlassSurface width="100%" height={48} borderRadius={14} onClick={continueInChat} className="cursor-pointer">
                  <span className="text-[#252525] font-semibold text-sm">Continue in chat</span>
                </GlassSurface>
                {!sessionId && (
                  <p className="text-xs text-[#252525]/50 mt-2 text-center">
                    Open this tool from chat to attach results to your conversation automatically.
                  </p>
                )}
              </div>
            </section>
          )}
        </section>
      </main>
    </div>
  )
}

function FieldSeverityFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-[#ebebeb] text-[#252525]/60 text-sm">
      Loading...
    </div>
  )
}

export default function FieldSeverityPage() {
  return (
    <Suspense fallback={<FieldSeverityFallback />}>
      <FieldSeverityContent />
    </Suspense>
  )
}
