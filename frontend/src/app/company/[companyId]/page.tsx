"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { fetchCompanyDetails, fetchRiskHistory, fetchCompanyNews } from "@/lib/api";
import {
  Building2, ArrowLeft, ShieldAlert, Activity, FileText, CheckCircle2, AlertOctagon, Newspaper, ExternalLink
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

export default function CompanyDetailPage() {
  const params = useParams();
  const companyId = Number(params.companyId);

  const [details, setDetails] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [news, setNews] = useState<any[]>([]);
  const [newsNotice, setNewsNotice] = useState<string | null>(null);
  const [sampleNews, setSampleNews] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadCompanyData() {
      try {
        setLoading(true);
        const [detData, histData, newsData] = await Promise.all([
          fetchCompanyDetails(companyId),
          fetchRiskHistory(companyId),
          fetchCompanyNews(companyId).catch(() => ({ articles: [] }))
        ]);
        setDetails(detData);
        setHistory(histData.risk_history || []);
        setNews(newsData.articles || []);
        setNewsNotice(newsData.notice || null);
        setSampleNews(newsData.sample_sp500_news || []);
      } catch (err: any) {
        setError(err.message || "Failed to load company details.");
      } finally {
        setLoading(false);
      }
    }
    if (companyId) {
      loadCompanyData();
    }
  }, [companyId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4">
        <div className="w-12 h-12 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
        <p className="text-slate-400 text-sm">Loading Firm {companyId} analytics...</p>
      </div>
    );
  }

  if (error || !details) {
    return (
      <div className="bg-red-950/40 border border-red-800/60 rounded-xl p-6 text-red-300 max-w-xl mx-auto my-12">
        <h3 className="font-bold text-lg mb-2">Error Loading Firm</h3>
        <p className="text-sm">{error || "Firm data could not be retrieved."}</p>
        <Link href="/companies" className="inline-block mt-4 text-xs font-semibold text-indigo-400 hover:underline">
          ← Return to Company Directory
        </Link>
      </div>
    );
  }

  const finRisk = details.financial?.financial_risk_score || 0;
  const newsRisk = 45.0;
  const finalRisk = details.risk_fusion?.final_multimodal_risk_score || finRisk;
  const category = details.risk_fusion?.risk_category || details.financial?.risk_category;

  const badgeStyle =
    category === "High Risk" || finRisk >= 60
      ? "bg-red-500/20 text-red-400 border-red-500/40"
      : category === "Medium Risk" || finRisk >= 30
      ? "bg-amber-500/20 text-amber-400 border-amber-500/40"
      : "bg-emerald-500/20 text-emerald-400 border-emerald-500/40";

  return (
    <div className="space-y-8">
      {/* Top Header */}
      <div>
        <Link href="/companies" className="inline-flex items-center space-x-1 text-xs text-slate-400 hover:text-slate-200 mb-4 transition-colors">
          <ArrowLeft className="h-4 w-4" />
          <span>Back to Company Directory</span>
        </Link>

        <div className="flex flex-col md:flex-row md:items-center justify-between bg-slate-900 border border-slate-800 p-6 rounded-2xl gap-4">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-indigo-600/20 text-indigo-400 rounded-xl border border-indigo-500/30">
              <Building2 className="h-8 w-8" />
            </div>
            <div>
              <div className="flex items-center space-x-3">
                <h1 className="text-2xl font-extrabold text-white">Firm {companyId}</h1>
                <span className={`text-xs px-3 py-1 rounded-full border font-semibold ${badgeStyle}`}>
                  {category}
                </span>
              </div>
              <p className="text-slate-400 text-xs mt-1">
                Recorded Panel Periods: {details.total_periods_recorded} | Latest Recorded Period: Time {details.latest_time_period}
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-6 bg-slate-950/60 p-4 rounded-xl border border-slate-800 text-center">
            <div>
              <span className="text-xs text-slate-400 block">Financial Risk</span>
              <span className="text-xl font-bold font-mono text-slate-200">{finRisk}</span>
            </div>
            <div className="h-8 w-px bg-slate-800"></div>
            <div>
              <span className="text-xs text-slate-400 block">News Risk</span>
              <span className="text-xl font-bold font-mono text-slate-400">{newsRisk}</span>
            </div>
            <div className="h-8 w-px bg-slate-800"></div>
            <div>
              <span className="text-xs text-indigo-400 block font-semibold">Final Multimodal Risk</span>
              <span className="text-2xl font-extrabold font-mono text-indigo-400">{finalRisk}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Temporal Risk History Chart */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm">
        <h3 className="font-bold text-slate-200 text-lg mb-1">Financial Distress Trajectory Time-Series</h3>
        <p className="text-xs text-slate-400 mb-6">Historical distress score trajectory over panel time steps (Time 1 to {details.latest_time_period}).</p>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time_period" stroke="#64748b" tickFormatter={(v) => `Time ${v}`} />
              <YAxis domain={[0, 100]} stroke="#64748b" />
              <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", color: "#f8fafc" }} />
              <Line type="monotone" dataKey="financial_risk_score" stroke="#818cf8" strokeWidth={3} dot={{ r: 4 }} name="Financial Distress Risk" />
              <Line type="monotone" dataKey="final_multimodal_risk" stroke="#6366f1" strokeDasharray="5 5" strokeWidth={2} name="Final Multimodal Risk" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Explainability & Multimodal Details */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Risk Contributors */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2">
            <AlertOctagon className="h-5 w-5 text-amber-400" />
            <h3 className="font-bold text-slate-200 text-base">Key Feature Contributors (XGBoost)</h3>
          </div>
          <p className="text-xs text-slate-400">
            {details.explainability?.disclaimer || "Features contributing to the model prediction according to XGBoost gain scores."}
          </p>

          <div className="space-y-3 pt-2">
            {details.financial?.top_contributing_features?.map((f: any) => (
              <div key={f.feature} className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/80 flex items-center justify-between">
                <div>
                  <span className="font-mono text-sm font-semibold text-indigo-300">{f.feature}</span>
                  <span className="text-xs text-slate-500 block">Value: {f.value}</span>
                </div>
                <div className="text-right">
                  <span className="text-xs font-semibold text-slate-300">Importance Gain</span>
                  <span className="block font-mono text-xs text-emerald-400">{(f.importance * 100).toFixed(2)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Multimodal Adaptive Fusion Details */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm space-y-4">
          <div className="flex items-center space-x-2">
            <Activity className="h-5 w-5 text-indigo-400" />
            <h3 className="font-bold text-slate-200 text-base">Multimodal Fusion Architecture</h3>
          </div>
          <p className="text-xs text-slate-400">
            Rule-based data-dependent adaptive risk fusion mathematical breakdown.
          </p>

          <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800 space-y-3 text-sm">
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Fusion Mechanism:</span>
              <span className="font-medium text-slate-200">{details.risk_fusion?.fusion_type}</span>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">Financial Model Weight (w_fin):</span>
              <span className="font-mono text-indigo-400 font-bold">{((details.risk_fusion?.weights?.financial_weight || 0.6) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between border-b border-slate-800/80 pb-2">
              <span className="text-slate-400">News Model Weight (w_news):</span>
              <span className="font-mono text-slate-300 font-bold">{((details.risk_fusion?.weights?.news_weight || 0.4) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">News Availability Factor (α):</span>
              <span className="font-mono text-slate-300">{details.risk_fusion?.availability_factor_alpha || 0.0}</span>
            </div>
          </div>

          <div className="p-4 bg-indigo-950/20 border border-indigo-900/40 rounded-xl text-xs text-indigo-300/80 leading-relaxed">
            <strong>Scientific Rigor Note:</strong> This assessment predicts financial distress probability based on financial ratios. Real historical news sentiment offers qualitative situational monitoring. Non-causal correlation only.
          </div>
        </div>
      </div>

      {/* Research Integrity Notice & Real News Stream Section */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-sm space-y-6">
        {newsNotice && (
          <div className="p-4 bg-slate-950 border border-indigo-900/40 rounded-xl space-y-2">
            <div className="flex items-center space-x-2 text-indigo-300 font-semibold text-sm">
              <ShieldAlert className="h-4 w-4 text-indigo-400" />
              <span>Entity Resolution & Research Integrity Standard</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              {newsNotice}
            </p>
          </div>
        )}

        <div>
          <div className="flex items-center space-x-2 mb-4">
            <Newspaper className="h-5 w-5 text-indigo-400" />
            <h3 className="font-bold text-slate-200 text-lg">
              {news.length > 0 ? "Firm Financial News Stream" : "Real S&P 500 Financial News Stream (NLP Sentiment Pipeline Demonstration)"}
            </h3>
          </div>

          <div className="space-y-3">
            {(news.length > 0 ? news : sampleNews).map((item: any, idx: number) => {
              const sentimentBadge =
                item.dominant_sentiment === "negative"
                  ? "bg-red-500/10 text-red-400 border-red-500/30"
                  : item.dominant_sentiment === "positive"
                  ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                  : "bg-slate-800 text-slate-400 border-slate-700";

              return (
                <div key={idx} className="bg-slate-950/60 p-4 rounded-xl border border-slate-800/80 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      {item.symbol && (
                        <span className="text-xs font-mono font-bold text-indigo-400 bg-indigo-950/40 px-2 py-0.5 rounded border border-indigo-900/60">
                          {item.symbol}
                        </span>
                      )}
                      <span className="text-xs text-slate-500 font-mono">{item.date}</span>
                      <span className={`text-xs px-2 py-0.5 rounded-full border font-semibold ${sentimentBadge}`}>
                        {item.dominant_sentiment?.toUpperCase()}
                      </span>
                    </div>
                    <p className="text-sm font-medium text-slate-200">{item.headline}</p>
                  </div>
                  <div className="text-right sm:self-center font-mono shrink-0">
                    <span className="text-xs text-slate-400 block">Sentiment Risk</span>
                    <span className="text-base font-bold text-indigo-300">{item.sentiment_risk_score} / 100</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
