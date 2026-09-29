"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchStatistics, fetchCompanies, fetchFeatureImportance } from "@/lib/api";
import {
  Building2, AlertTriangle, ShieldCheck, Activity, Newspaper, ChevronRight, CheckCircle2
} from "lucide-react";
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid
} from "recharts";

export default function Dashboard() {
  const [stats, setStats] = useState<any>(null);
  const [companies, setCompanies] = useState<any[]>([]);
  const [features, setFeatures] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadDashboardData() {
      try {
        setLoading(true);
        const [statsData, companiesData, featureData] = await Promise.all([
          fetchStatistics(),
          fetchCompanies(1, 10),
          fetchFeatureImportance()
        ]);
        setStats(statsData);
        setCompanies(companiesData.companies || []);
        setFeatures((featureData.top_features || []).slice(0, 8));
      } catch (err: any) {
        setError(err.message || "Failed to load backend data. Ensure FastAPI server is running.");
      } finally {
        setLoading(false);
      }
    }
    loadDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm">Loading Multimodal Financial Risk Platform...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-6 text-red-300 max-w-2xl mx-auto my-12">
        <div className="flex items-center space-x-3 mb-2">
          <AlertTriangle className="h-6 w-6 text-red-400" />
          <h3 className="font-bold text-lg">Backend API Offline</h3>
        </div>
        <p className="text-sm text-red-200/80 mb-4">{error}</p>
        <p className="text-xs bg-slate-900/80 p-3 rounded font-mono text-slate-300">
          Make sure FastAPI is running locally: uvicorn backend.main:app --port 8000
        </p>
      </div>
    );
  }

  const pieData = [
    { name: "Low Risk", value: stats?.low_risk_companies || 0, color: "#10b981" },
    { name: "Medium Risk", value: stats?.medium_risk_companies || 0, color: "#f59e0b" },
    { name: "High Risk", value: stats?.high_risk_companies || 0, color: "#ef4444" },
  ];

  return (
    <div className="space-y-8">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 p-6 rounded-2xl border border-indigo-900/40 shadow-xl">
        <div className="inline-block bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-xs px-3 py-1 rounded-full mb-2 font-medium">
          Scopus-Oriented Research Implementation
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
          Dynamic Multimodal Financial Risk Assessment
        </h1>
        <p className="text-slate-400 text-sm mt-1 max-w-3xl">
          Integrating XGBoost financial distress classification with FinBERT financial news sentiment and 90-day PyTorch temporal LSTM sequence modeling.
        </p>
      </div>

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Financial Firms</span>
            <div className="p-2 bg-indigo-500/10 rounded-lg text-indigo-400">
              <Building2 className="h-5 w-5" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white mt-2">{stats?.total_companies || 422}</div>
          <span className="text-xs text-slate-500 mt-1 block">3,672 panel observations</span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Distressed Records</span>
            <div className="p-2 bg-amber-500/10 rounded-lg text-amber-400">
              <AlertTriangle className="h-5 w-5" />
            </div>
          </div>
          <div className="text-2xl font-bold text-amber-400 mt-2">{stats?.financially_distressed_records || 136}</div>
          <span className="text-xs text-slate-500 mt-1 block">3.70% baseline class ratio</span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Real News Articles</span>
            <div className="p-2 bg-purple-500/10 rounded-lg text-purple-400">
              <Newspaper className="h-5 w-5" />
            </div>
          </div>
          <div className="text-2xl font-bold text-purple-400 mt-2">{stats?.real_news_articles_count || 4589}</div>
          <span className="text-xs text-slate-500 mt-1 block">469 S&P 500 corporate entities</span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Avg Financial Risk</span>
            <div className="p-2 bg-emerald-500/10 rounded-lg text-emerald-400">
              <Activity className="h-5 w-5" />
            </div>
          </div>
          <div className="text-2xl font-bold text-white mt-2">{stats?.average_financial_risk_score || 0} / 100</div>
          <span className="text-xs text-emerald-400 mt-1 block">Standardized probability score</span>
        </div>

        <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">High Risk Alert</span>
            <div className="p-2 bg-red-500/10 rounded-lg text-red-400">
              <ShieldCheck className="h-5 w-5" />
            </div>
          </div>
          <div className="text-2xl font-bold text-red-400 mt-2">{stats?.high_risk_companies || 0}</div>
          <span className="text-xs text-slate-500 mt-1 block">Score ≥ 60 threshold</span>
        </div>
      </div>

      {/* Visualizations Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Risk Distribution Donut */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm flex flex-col">
          <h3 className="font-bold text-slate-200 text-base mb-1">Company Risk Category Distribution</h3>
          <p className="text-xs text-slate-400 mb-4">Proportion of companies categorized by multimodal risk level.</p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", color: "#f8fafc" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-6 mt-2 text-xs">
            {pieData.map((item) => (
              <div key={item.name} className="flex items-center space-x-2">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></span>
                <span className="text-slate-300">{item.name}: {item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Feature Importance Bar Chart */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm flex flex-col">
          <h3 className="font-bold text-slate-200 text-base mb-1">Top XGBoost Financial Feature Contributors</h3>
          <p className="text-xs text-slate-400 mb-4">Relative gain importance of top financial ratio variables.</p>
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={features} layout="vertical" margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis type="number" stroke="#64748b" tick={{ fontSize: 12 }} />
                <YAxis dataKey="feature" type="category" stroke="#64748b" tick={{ fontSize: 12 }} width={50} />
                <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", color: "#f8fafc" }} />
                <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Companies Table Preview */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="font-bold text-slate-200 text-lg">Companies Risk Directory Preview</h3>
            <p className="text-xs text-slate-400">Latest financial distress and multimodal risk assessments.</p>
          </div>
          <Link
            href="/companies"
            className="flex items-center space-x-1 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
          >
            <span>View All Companies</span>
            <ChevronRight className="h-4 w-4" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                <th className="py-3 px-4">Firm ID</th>
                <th className="py-3 px-4">Historical Distress</th>
                <th className="py-3 px-4">Financial Risk</th>
                <th className="py-3 px-4">News Risk</th>
                <th className="py-3 px-4">Final Risk</th>
                <th className="py-3 px-4">Risk Level</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {companies.map((comp) => {
                const badgeColor =
                  comp.risk_category === "High Financial Risk" || comp.financial_risk_score >= 60
                    ? "bg-red-500/10 text-red-400 border-red-500/30"
                    : comp.risk_category === "Medium Financial Risk" || comp.financial_risk_score >= 30
                    ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                    : "bg-emerald-500/10 text-emerald-400 border-emerald-500/30";

                return (
                  <tr key={comp.company_id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="py-3 px-4 font-semibold text-slate-200">Firm {comp.company_id}</td>
                    <td className="py-3 px-4">
                      {comp.is_distressed_historical === 1 ? (
                        <span className="text-xs bg-red-950 text-red-300 px-2 py-0.5 rounded">Distressed</span>
                      ) : (
                        <span className="text-xs bg-slate-800 text-slate-400 px-2 py-0.5 rounded">Healthy</span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono font-medium text-slate-200">{comp.financial_risk_score}</td>
                    <td className="py-3 px-4 font-mono text-slate-400">{comp.news_risk_score}</td>
                    <td className="py-3 px-4 font-mono font-bold text-indigo-400">{comp.final_risk_score}</td>
                    <td className="py-3 px-4">
                      <span className={`text-xs px-2.5 py-1 rounded-full border font-medium ${badgeColor}`}>
                        {comp.risk_category}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <Link
                        href={`/company/${comp.company_id}`}
                        className="text-xs text-indigo-400 hover:text-indigo-300 font-medium hover:underline"
                      >
                        Inspect →
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
