import type { Metadata, Viewport } from "next";
import { Newsreader, Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { ThemeProvider, THEME_INIT_SCRIPT } from "@/lib/theme";
import { PersonalizationProvider } from "@/lib/personalization";
import { Header } from "@/components/Header";
import { BottomNav } from "@/components/BottomNav";
import { AmbientField } from "@/components/AmbientField";
import { PwaStatus } from "@/components/PwaStatus";

const newsreader = Newsreader({
  variable: "--font-newsreader",
  subsets: ["latin"],
  style: ["normal", "italic"],
  weight: ["400", "500", "600"],
});

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const jbMono = JetBrains_Mono({
  variable: "--font-jbmono",
  subsets: ["latin"],
  weight: ["400", "500"],
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
  themeColor: [
    // eslint-disable-next-line no-restricted-syntax -- see comment above
    { media: "(prefers-color-scheme: light)", color: "#f7f5ef" },
    // eslint-disable-next-line no-restricted-syntax -- see comment above
    { media: "(prefers-color-scheme: dark)", color: "#131319" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
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
      <body className={`${newsreader.variable} ${inter.variable} ${jbMono.variable}`}>
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
