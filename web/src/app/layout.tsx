import type { Metadata, Viewport } from "next";
import { Fraunces, Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider, THEME_INIT_SCRIPT } from "@/lib/theme";
import { PersonalizationProvider } from "@/lib/personalization";
import { Header } from "@/components/Header";
import { BottomNav } from "@/components/BottomNav";
import { AmbientField } from "@/components/AmbientField";
import { PwaStatus } from "@/components/PwaStatus";

// EDITORIAL register — headlines, summaries, analysis.
//
// `axes` is load-bearing here, not decoration: next/font/google ships ONLY
// the wght axis for a variable font by default (verified in
// node_modules/next/dist/docs/01-app/03-api-reference/02-components/font.md
// §axes), so without this list every `font-variation-settings: "opsz" ...`
// in the stylesheets would silently do nothing. opsz is what lets one
// family read as a display face at 2.2rem and as a text face at 1rem;
// SOFT/WONK carry the character that keeps it from looking like stock
// Georgia. No `weight` — supplying one would pin it to a static instance
// and defeat the variable axes.
const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
  axes: ["SOFT", "WONK", "opsz"],
});

// UI register — controls, body copy, anything not a headline or a metric.
const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
});

// INSTRUMENT register — rank, score, band, category, source, counts,
// timestamps. Geist Mono is designed as a pair with Geist above, which is
// why the metadata layer reads as engineered rather than assembled.
const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Observatory — AI News",
  description: "An importance-ranked, on-device-personalized feed of AI news.",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: [
      { url: "/icons/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/icons/favicon-16.png", sizes: "16x16", type: "image/png" },
    ],
    apple: [{ url: "/icons/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  appleWebApp: {
    capable: true,
    title: "Observatory",
    statusBarStyle: "black-translucent",
  },
};

// The theme-color meta tag has to be a literal, static value per scheme —
// it's read by the OS/browser chrome before any CSS has loaded, so it
// cannot reference a CSS custom property the way every other colour in
// this app does. These two values are kept in sync by hand with
// globals.css's --color-surface (light) and dark data-theme block; there
// is nothing to re-derive them from at this layer.
export const viewport: Viewport = {
  // viewport-fit=cover. Without this, iOS reports every
  // env(safe-area-inset-*) as 0, which silently defeated FIVE call sites
  // that depend on it: globals.css's .appShell > main bottom padding,
  // BottomNav.module.css (x2), PwaStatus.module.css, and
  // StoryDeck.module.css. Combined with appleWebApp.statusBarStyle
  // "black-translucent" above -- which deliberately extends content under
  // the status bar -- the installed iPhone PWA was drawing its floating
  // nav inside the home-indicator zone. The insets were always written
  // correctly; they were just never switched on.
  viewportFit: "cover",
  themeColor: [
    // eslint-disable-next-line no-restricted-syntax -- see comment above
    { media: "(prefers-color-scheme: light)", color: "#f4f4f0" },
    // eslint-disable-next-line no-restricted-syntax -- see comment above
    { media: "(prefers-color-scheme: dark)", color: "#0c0c11" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // The three next/font `.variable` classes belong on <html>, NOT <body>.
    // globals.css builds --font-serif/--font-sans/--font-mono at :root from
    // var(--font-fraunces) etc; a custom property set on <body> is invisible
    // to a rule on :root, because custom properties inherit downward only.
    // With them on <body> the :root declarations were invalid at
    // computed-value time, so --font-serif/-sans/-mono inherited as
    // guaranteed-invalid and EVERY font-family in the app silently fell back
    // to the browser default -- measured: the whole UI rendered in Times New
    // Roman, on this build and on HEAD before it. Moving them up one element
    // is the entire fix.
    <html
      lang="en"
      suppressHydrationWarning
      className={`${fraunces.variable} ${geist.variable} ${geistMono.variable}`}
    >
      <head>
        {/* Set data-theme before paint to avoid a flash of the wrong theme. */}
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
        {/* The bundle (spec §4.5: "fetches the manifest and bundle at
            runtime", never baked in at build time) and the lead-image
            resizing proxy (see LeadImage.tsx) are both third-party origins
            every page load depends on. Warming the connection in parallel
            with the JS bundle loading, instead of only starting DNS/TCP/TLS
            once a fetch() call actually needs it, removes that handshake
            from the critical path entirely. */}
        <link rel="preconnect" href="https://raw.githubusercontent.com" />
        <link rel="preconnect" href="https://wsrv.nl" />
        <link rel="dns-prefetch" href="https://raw.githubusercontent.com" />
        <link rel="dns-prefetch" href="https://wsrv.nl" />
      </head>
      <body>
        <ThemeProvider>
          <PersonalizationProvider>
            <AmbientField />
            <div className="appShell">
              <Header />
              <PwaStatus />
              <main>{children}</main>
              <BottomNav />
            </div>
          </PersonalizationProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
