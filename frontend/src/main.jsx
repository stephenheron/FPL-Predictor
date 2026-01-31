import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './style.css'

const positionOrder = ['GK', 'DEF', 'MID', 'FWD']
const positionLabels = {
  GK: 'Goalkeepers',
  DEF: 'Defenders',
  MID: 'Midfielders',
  FWD: 'Forwards',
}

const positionBadge = {
  GK: 'bg-emerald-400/10 text-emerald-200 ring-emerald-400/30',
  DEF: 'bg-sky-400/10 text-sky-200 ring-sky-400/30',
  MID: 'bg-amber-400/10 text-amber-200 ring-amber-400/30',
  FWD: 'bg-rose-400/10 text-rose-200 ring-rose-400/30',
}

const formatNumber = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(value)) return '-'
  return Number(value).toFixed(digits)
}

const groupByPosition = (players) => {
  const grouped = {
    GK: [],
    DEF: [],
    MID: [],
    FWD: [],
  }

  players.forEach((player) => {
    if (grouped[player.position]) {
      grouped[player.position].push(player)
    }
  })

  return grouped
}

const getPlayerKey = (player) =>
  player.player_id ?? player.fpl_id ?? `${player.name}-${player.team}`

const PlayerCard = ({ player }) => {
  const badgeClass =
    positionBadge[player.position] || 'bg-slate-400/10 text-slate-200 ring-slate-400/30'

  return (
    <article className="rounded-2xl border border-white/10 bg-slate-900/70 p-4 shadow-lg shadow-black/20">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="text-lg font-semibold text-white">{player.name}</h4>
          <p className="text-sm text-slate-300">
            {player.team} vs {player.opponent_name}
          </p>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${badgeClass}`}
        >
          {player.position}
        </span>
      </div>
      <div className="mt-4 grid grid-cols-1 gap-3 text-sm">
        <div>
          <p className="text-slate-400">Predicted</p>
          <p className="text-lg font-semibold text-white">
            {formatNumber(player.predicted_points, 1)}
          </p>
        </div>
      </div>
    </article>
  )
}

const PositionGroup = ({ players, title }) => {
  if (!players.length) return null

  return (
    <section className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">{title}</h3>
        <span className="text-sm text-slate-400">{players.length} players</span>
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {players.map((player) => (
          <PlayerCard key={getPlayerKey(player)} player={player} />
        ))}
      </div>
    </section>
  )
}

const PitchCard = ({ player }) => {
  const badgeClass =
    positionBadge[player.position] || 'bg-slate-400/10 text-slate-200 ring-slate-400/30'

  return (
    <article className="w-36 rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-3 text-center shadow-lg shadow-black/30 backdrop-blur">
      <div className="mb-2 flex items-center justify-center">
        <span
          className={`inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] ring-1 ${badgeClass}`}
        >
          {player.position}
        </span>
      </div>
      <p className="text-sm font-semibold text-white">{player.name}</p>
      <p className="text-xs text-slate-300">{player.team}</p>
      <p className="mt-2 text-lg font-semibold text-white">
        {formatNumber(player.predicted_points, 1)}
      </p>
    </article>
  )
}

const PitchRow = ({ players }) => {
  if (!players.length) return null

  return (
    <div className="flex flex-wrap items-center justify-center gap-4">
      {players.map((player) => (
        <PitchCard key={getPlayerKey(player)} player={player} />
      ))}
    </div>
  )
}

const Pitch = ({ players }) => {
  const grouped = useMemo(() => groupByPosition(players), [players])

  return (
    <section className="relative overflow-hidden rounded-3xl border border-emerald-200/20 bg-gradient-to-b from-emerald-900/70 via-emerald-900/40 to-emerald-950/80 p-6">
      <div className="pointer-events-none absolute inset-6 rounded-2xl border border-emerald-200/20"></div>
      <div className="pointer-events-none absolute left-1/2 top-6 bottom-6 w-px -translate-x-1/2 bg-emerald-200/20"></div>
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-28 w-28 -translate-x-1/2 -translate-y-1/2 rounded-full border border-emerald-200/20"></div>
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-200/30"></div>

      <div className="relative grid gap-6">
        <PitchRow players={grouped.GK} />
        <PitchRow players={grouped.DEF} />
        <PitchRow players={grouped.MID} />
        <PitchRow players={grouped.FWD} />
      </div>
    </section>
  )
}

const BenchSection = ({ players }) => {
  const grouped = useMemo(() => groupByPosition(players), [players])

  return (
    <section className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-white">Bench</h2>
        <span className="text-sm text-slate-400">Substitutes</span>
      </div>
      {positionOrder.map((position) => (
        <PositionGroup
          key={position}
          players={grouped[position]}
          title={positionLabels[position]}
        />
      ))}
    </section>
  )
}

const ErrorView = ({ message }) => (
  <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
    <div className="mx-auto max-w-3xl rounded-3xl border border-red-500/40 bg-red-950/40 p-8">
      <h1 className="text-2xl font-semibold text-white">Unable to load squad</h1>
      <p className="mt-3 text-sm text-red-200">{message}</p>
    </div>
  </main>
)

const LoadingView = () => (
  <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
    <div className="mx-auto max-w-3xl rounded-3xl border border-white/10 bg-slate-900/70 p-8">
      <p className="text-sm uppercase tracking-[0.25em] text-slate-400">FPL Predictions</p>
      <h1 className="mt-3 text-3xl font-semibold text-white">Loading optimal squad...</h1>
      <div className="mt-6 h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div className="h-full w-1/3 animate-pulse rounded-full bg-slate-500"></div>
      </div>
    </div>
  </main>
)

const App = () => {
  const [status, setStatus] = useState('loading')
  const [data, setData] = useState(null)

  useEffect(() => {
    let active = true

    const loadSquad = async () => {
      setStatus('loading')

      try {
        const response = await fetch('data/optimal_squad.json')
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const payload = await response.json()
        if (!active) return
        setData(payload)
        setStatus('ready')
      } catch (error) {
        if (!active) return
        setStatus('error')
      }
    }

    loadSquad()

    return () => {
      active = false
    }
  }, [])

  if (status === 'loading') {
    return <LoadingView />
  }

  if (status === 'error') {
    return (
      <ErrorView message="Make sure the data file exists at /public/data/optimal_squad.json and is included in the deployed build output." />
    )
  }

  if (!data || !Array.isArray(data.players)) {
    return <ErrorView message="The squad data response is missing a players array." />
  }

  const starters = data.players.filter((player) => player.is_starter)
  const bench = data.players.filter((player) => !player.is_starter)

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
      <div className="mx-auto max-w-6xl space-y-10">
        <header className="space-y-4">
          <p className="text-sm uppercase tracking-[0.25em] text-slate-400">FPL Predictions</p>
          <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <h1 className="text-4xl font-semibold text-white">Optimal Squad Overview</h1>
          </div>
        </header>

        <div className="grid gap-8">
          <Pitch players={starters} />
          <BenchSection players={bench} />
        </div>
      </div>
    </main>
  )
}

const root = createRoot(document.querySelector('#app'))
root.render(<App />)
