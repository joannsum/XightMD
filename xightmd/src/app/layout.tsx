import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'XightMD - AI-Powered Chest X-ray Analysis',
  description: 'Multi-agent AI system for radiologists to analyze chest X-rays and generate structured reports using Claude 4 and Fetch.ai uAgents.',
  keywords: ['AI', 'radiology', 'chest X-ray', 'medical imaging', 'Claude', 'uAgents'],
  authors: [{ name: 'XightMD Team' }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} antialiased h-full bg-gray-50`}>
        {children}
      </body>
    </html>
  );
}