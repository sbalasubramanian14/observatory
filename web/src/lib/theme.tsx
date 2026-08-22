// Light/dark/system theming (spec §4.5 "Theming"). System is the default
// and follows the OS live. The resolved theme (light|dark) is written to
// `data-theme` on <html>; every colour in the app is a CSS custom property
// keyed off that attribute (see globals.css) — components never hold a
// colour value themselves.
"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useSyncExternalStore } from "react";

export type ThemePreference = "light" | "dark" | "system";
type ResolvedTheme = "light" | "dark";

const STORAGE_KEY = "observatory:theme";
// localStorage's own `storage` event only fires in *other* tabs, never the
// tab that made the write, so a same-tab UI update needs its own signal.
// This custom event fills that gap for useSyncExternalStore below.
const LOCAL_SYNC_EVENT = "observatory:theme-sync";

function getSystemTheme(): ResolvedTheme {
  if (typeof window === "undefined") return "light";
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function readStoredPreference(): ThemePreference {
  if (typeof window === "undefined") return "system";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark" || stored === "system") return stored;
  return "system";
}

function subscribePreference(callback: () => void) {
  window.addEventListener("storage", callback);
  window.addEventListener(LOCAL_SYNC_EVENT, callback);
  return () => {
    window.removeEventListener("storage", callback);
    window.removeEventListener(LOCAL_SYNC_EVENT, callback);
  };
}

function getServerPreferenceSnapshot(): ThemePreference {
  return "system";
}

function subscribeSystemTheme(callback: () => void) {
  const mql = window.matchMedia("(prefers-color-scheme: dark)");
  mql.addEventListener("change", callback);
  return () => mql.removeEventListener("change", callback);
}

function getServerSystemSnapshot(): ResolvedTheme {
  return "light";
}

function writePreference(pref: ThemePreference) {
  window.localStorage.setItem(STORAGE_KEY, pref);
  window.dispatchEvent(new Event(LOCAL_SYNC_EVENT));
}

interface ThemeContextValue {
  preference: ThemePreference;
  resolved: ResolvedTheme;
  setPreference: (pref: ThemePreference) => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  // useSyncExternalStore (not useState+useEffect) is the correct tool here:
  // it reads localStorage/matchMedia directly, gives a safe server snapshot
  // for the static export build, and re-renders on change without ever
  // calling setState from inside an effect.
  const preference = useSyncExternalStore(
    subscribePreference,
    readStoredPreference,
    getServerPreferenceSnapshot,
  );
  const systemTheme = useSyncExternalStore(
    subscribeSystemTheme,
    getSystemTheme,
    getServerSystemSnapshot,
  );
  const resolved: ResolvedTheme = preference === "system" ? systemTheme : preference;

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", resolved);
  }, [resolved]);

  const setPreference = useCallback((pref: ThemePreference) => {
    writePreference(pref);
  }, []);

  const value = useMemo(
    () => ({ preference, resolved, setPreference }),
    [preference, resolved, setPreference],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used within a ThemeProvider");
  return ctx;
}

/** Inline script string injected before hydration so the resolved theme is
 * correct on first paint (no flash of the wrong theme). */
export const THEME_INIT_SCRIPT = `
(function () {
  try {
    var stored = window.localStorage.getItem('${STORAGE_KEY}');
    var pref = stored === 'light' || stored === 'dark' || stored === 'system' ? stored : 'system';
    var resolved = pref === 'system'
      ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
      : pref;
    document.documentElement.setAttribute('data-theme', resolved);
  } catch (e) {}
})();
`;
