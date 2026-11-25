import { API_BASE_URL } from '../config'

export function getCookie(name) {
  const prefix = name + '='
  for (const part of document.cookie.split(';')) {
    const cookie = part.trim()
    if (cookie.startsWith(prefix)) return cookie.slice(prefix.length)
  }
  return null
}

export function setCookie(name, value, days) {
  const expires = new Date(Date.now() + days * 24 * 60 * 60 * 1000)
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/;SameSite=Strict`
}

export function clearAuth() {
  document.cookie = 'auth_token=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;'
  document.cookie = 'user_email=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;'
}

export function authToken() {
  return getCookie('auth_token') || ''
}

export async function apiFetch(path, options = {}) {
  const headers = { ...(options.headers || {}) }
  const token = authToken()
  if (token) headers.Authorization = `Bearer ${token}`
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json'
  }
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers })
  if (response.status === 401) {
    clearAuth()
    window.location.href = '/'
  }
  return response
}

export async function apiGet(path) {
  const response = await apiFetch(path)
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json()
}

export async function apiSend(path, method, body) {
  const response = await apiFetch(path, {
    method,
    body: body instanceof FormData ? body : JSON.stringify(body),
  })
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json()
}

async function errorMessage(response) {
  try {
    const data = await response.json()
    return data.error || data.message || `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

export function formatCurrency(value, currency = 'EUR') {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
    maximumFractionDigits: 2,
  }).format(Number(value) || 0)
}

export function formatPercent(value) {
  const number = Number(value) || 0
  return `${number > 0 ? '+' : ''}${number.toFixed(2)}%`
}

export function changeColor(value) {
  return Number(value) > 0 ? 'success' : Number(value) < 0 ? 'error' : 'grey'
}
