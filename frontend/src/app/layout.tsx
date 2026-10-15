import type { Metadata, Viewport } from "next";
import "./globals.css";

export const viewport: Viewport = {
  themeColor: "#0b0d13",
};

export const metadata: Metadata = {
  title: "CardioPulse AI — Pre-Arrest Cardiac Warning System",
  description:
    "Digitize standard ECG waveform plots into 1D time-series, extract 90 HRV physiological biomarkers, and detect pre-arrest cardiac patterns with high precision.",
  icons: {
    icon: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://api.fontshare.com" />
        <link
          href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,600,700,900&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
