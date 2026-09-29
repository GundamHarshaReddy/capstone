"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck, Activity, BarChart3, Sliders, Layers, BookOpen } from "lucide-react";

export default function Navbar() {
  const pathname = usePathname();

  const navItems = [
    { label: "Dashboard", href: "/", icon: Activity },
    { label: "Company Directory", href: "/companies", icon: Layers },
    { label: "Risk Sandbox", href: "/predict", icon: Sliders },
    { label: "Experiments & Ablation", href: "/performance", icon: BarChart3 },
  ];

  return (
    <nav className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50 backdrop-blur bg-opacity-90">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-indigo-600 rounded-lg text-white shadow-lg shadow-indigo-500/30">
              <ShieldCheck className="h-6 w-6" />
            </div>
            <div>
              <span className="font-bold text-base sm:text-lg text-white tracking-tight">
                FinRisk Multimodal AI
              </span>
              <span className="block text-xs text-indigo-400 font-medium">
                Financial Indicators & Real Financial News Intelligence
              </span>
            </div>
          </div>
          <div className="flex space-x-1 sm:space-x-3">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-md text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? "bg-indigo-600/20 text-indigo-400 border border-indigo-500/30"
                      : "text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
