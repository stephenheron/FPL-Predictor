import './style.css'

const app = document.querySelector('#app')

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

const renderPlayerCard = (player) => {
  const badgeClass = positionBadge[player.position] || 'bg-slate-400/10 text-slate-200 ring-slate-400/30'

  return `
    <article class="rounded-2xl border border-white/10 bg-slate-900/70 p-4 shadow-lg shadow-black/20">
      <div class="flex items-start justify-between gap-3">
        <div>
          <h4 class="text-lg font-semibold text-white">${player.name}</h4>
          <p class="text-sm text-slate-300">${player.team} vs ${player.opponent_name}</p>
        </div>
        <span class="inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${badgeClass}">
          ${player.position}
        </span>
      </div>
      <div class="mt-4 grid grid-cols-1 gap-3 text-sm">
        <div>
          <p class="text-slate-400">Predicted</p>
          <p class="text-lg font-semibold text-white">${formatNumber(player.predicted_points, 1)}</p>
        </div>
      </div>
    </article>
  `
}

const renderPositionGroup = (players, title) => {
  if (!players.length) return ''

  return `
    <section class="space-y-4">
      <div class="flex items-center justify-between">
        <h3 class="text-xl font-semibold text-white">${title}</h3>
        <span class="text-sm text-slate-400">${players.length} players</span>
      </div>
      <div class="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        ${players.map(renderPlayerCard).join('')}
      </div>
    </section>
  `
}

const renderPitchCard = (player) => {
  const badgeClass = positionBadge[player.position] || 'bg-slate-400/10 text-slate-200 ring-slate-400/30'

  return `
    <article class="w-36 rounded-2xl border border-white/10 bg-slate-950/70 px-3 py-3 text-center shadow-lg shadow-black/30 backdrop-blur">
      <div class="mb-2 flex items-center justify-center">
        <span class="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.2em] ring-1 ${badgeClass}">
          ${player.position}
        </span>
      </div>
      <p class="text-sm font-semibold text-white">${player.name}</p>
      <p class="text-xs text-slate-300">${player.team}</p>
      <p class="mt-2 text-lg font-semibold text-white">${formatNumber(player.predicted_points, 1)}</p>
    </article>
  `
}

const renderPitchRow = (players) => {
  if (!players.length) return ''

  return `
    <div class="flex flex-wrap items-center justify-center gap-4">
      ${players.map(renderPitchCard).join('')}
    </div>
  `
}

const renderPitch = (players) => {
  const grouped = groupByPosition(players)

  return `
    <section class="relative overflow-hidden rounded-3xl border border-emerald-200/20 bg-gradient-to-b from-emerald-900/70 via-emerald-900/40 to-emerald-950/80 p-6">
      <div class="pointer-events-none absolute inset-6 rounded-2xl border border-emerald-200/20"></div>
      <div class="pointer-events-none absolute left-1/2 top-6 bottom-6 w-px -translate-x-1/2 bg-emerald-200/20"></div>
      <div class="pointer-events-none absolute left-1/2 top-1/2 h-28 w-28 -translate-x-1/2 -translate-y-1/2 rounded-full border border-emerald-200/20"></div>
      <div class="pointer-events-none absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-emerald-200/30"></div>

      <div class="relative grid gap-6">
        ${renderPitchRow(grouped.GK)}
        ${renderPitchRow(grouped.DEF)}
        ${renderPitchRow(grouped.MID)}
        ${renderPitchRow(grouped.FWD)}
      </div>
    </section>
  `
}

const renderBenchSection = (players) => {
  const grouped = groupByPosition(players)
  const sections = positionOrder
    .map((position) => renderPositionGroup(grouped[position], positionLabels[position]))
    .join('')

  return `
    <section class="space-y-6">
      <div class="flex items-center justify-between">
        <h2 class="text-2xl font-semibold text-white">Bench</h2>
        <span class="text-sm text-slate-400">Substitutes</span>
      </div>
      ${sections}
    </section>
  `
}

const renderError = (message) => {
  app.innerHTML = `
    <main class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
      <div class="mx-auto max-w-3xl rounded-3xl border border-red-500/40 bg-red-950/40 p-8">
        <h1 class="text-2xl font-semibold text-white">Unable to load squad</h1>
        <p class="mt-3 text-sm text-red-200">${message}</p>
      </div>
    </main>
  `
}

const renderApp = (data) => {
  const starters = data.players.filter((player) => player.is_starter)
  const bench = data.players.filter((player) => !player.is_starter)

  app.innerHTML = `
    <main class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
      <div class="mx-auto max-w-6xl space-y-10">
        <header class="space-y-4">
          <p class="text-sm uppercase tracking-[0.25em] text-slate-400">FPL Predictions</p>
          <div class="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
            <h1 class="text-4xl font-semibold text-white">Optimal Squad Overview</h1>
          </div>
        </header>

        <div class="grid gap-8">
          ${renderPitch(starters)}
          ${renderBenchSection(bench)}
        </div>
      </div>
    </main>
  `
}

const renderLoading = () => {
  app.innerHTML = `
    <main class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-12">
      <div class="mx-auto max-w-3xl rounded-3xl border border-white/10 bg-slate-900/70 p-8">
        <p class="text-sm uppercase tracking-[0.25em] text-slate-400">FPL Predictions</p>
        <h1 class="mt-3 text-3xl font-semibold text-white">Loading optimal squad...</h1>
        <div class="mt-6 h-2 w-full overflow-hidden rounded-full bg-slate-800">
          <div class="h-full w-1/3 animate-pulse rounded-full bg-slate-500"></div>
        </div>
      </div>
    </main>
  `
}

const loadSquad = async () => {
  renderLoading()

  try {
    const response = await fetch('data/optimal_squad.json')
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const data = await response.json()
    renderApp(data)
  } catch (error) {
    renderError('Make sure the data file exists at /public/data/optimal_squad.json and is included in the deployed build output.')
  }
}

loadSquad()
