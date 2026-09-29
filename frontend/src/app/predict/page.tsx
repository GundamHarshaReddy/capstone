"use client";

import { useState } from "react";
import { predictFinalRisk } from "@/lib/api";
import { Sliders, Play, Activity, ShieldAlert, Sparkles } from "lucide-react";

export default function PredictSandboxPage() {
  const [headline, setHeadline] = useState(
    "Supplier reports supply chain disruptions and revenue decline due to liquidity issues."
  );
  const [articleCount, setArticleCount] = useState(5);
  const [volatility, setVolatility] = useState(1.2);
  const [x36, setX36] = useState(1.0);
  const [x48, setX48] = useState(0.8);
  const [x25, setX25] = useState(0.5);

  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const payload = {
        financial_features: {
          x36: Number(x36),
          x48: Number(x48),
          x25: Number(x25),
        },
        news_headline: headline,
        article_count: Number(articleCount),
        sentiment_volatility: Number(volatility),
      };
      const res = await predictFinalRisk(payload);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to calculate prediction.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
        <h1 className="text-2xl font-bold text-white tracking-tight">Interactive Multimodal Risk Sandbox</h1>
        <p className="text-slate-400 text-sm mt-1">
          Simulate real-time financial distress predictions combined with FinBERT news sentiment and adaptive risk fusion.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form Inputs */}
        <form onSubmit={handlePredict} className="bg-slate-900/80 border border-slate-800 p-6 rounded-xl space-y-6">
          <div className="space-y-4">
            <h3 className="font-bold text-slate-200 text-base border-b border-slate-800 pb-2">Financial Feature Inputs</h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="text-xs text-slate-400 font-mono block mb-1">Feature x36</label>
                <input
                  type="number"
                  step="0.1"
                  value={x36}
                  onChange={(e) => setX36(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 font-mono block mb-1">Feature x48</label>
                <input
                  type="number"
                  step="0.1"
                  value={x48}
                  onChange={(e) => setX48(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 font-mono block mb-1">Feature x25</label>
                <input
                  type="number"
                  step="0.1"
                  value={x25}
                  onChange={(e) => setX25(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
                />
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="font-bold text-slate-200 text-base border-b border-slate-800 pb-2">News Sentiment & Volume Inputs</h3>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Recent Company Headline</label>
              <textarea
                rows={3}
                value={headline}
                onChange={(e) => setHeadline(e.target.value)}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-slate-400 block mb-1">Article Volume (90d)</label>
                <input
                  type="number"
                  min="0"
                  max="50"
                  value={articleCount}
                  onChange={(e) => setArticleCount(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
                />
              </div>
              <div>
                <label className="text-xs text-slate-400 block mb-1">Sentiment Volatility</label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  value={volatility}
                  onChange={(e) => setVolatility(Number(e.target.value))}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200"
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center space-x-2 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 rounded-xl shadow-lg transition-colors"
          >
            {loading ? (
              <span>Calculating Model Risk Scores...</span>
            ) : (
              <>
                <Play className="h-4 w-4" />
                <span>Execute Multimodal Risk Prediction</span>
              </>
            )}
          </button>
        </form>

        {/* Prediction Results */}
        <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-xl flex flex-col justify-between">
          <div>
            <h3 className="font-bold text-slate-200 text-lg mb-4 flex items-center space-x-2">
              <Sparkles className="h-5 w-5 text-indigo-400" />
              <span>Real-Time Model Output</span>
            </h3>

            {error && (
              <div className="bg-red-950/40 border border-red-800 text-red-300 p-4 rounded-lg text-sm mb-4">
                {error}
              </div>
            )}

            {result ? (
              <div className="space-y-6">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-xs text-slate-400 block font-semibold">Financial Risk</span>
                    <span className="text-2xl font-bold font-mono text-slate-200 mt-1 block">
                      {result.financial_prediction?.financial_risk_score}
                    </span>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <span className="text-xs text-slate-400 block font-semibold">News Risk</span>
                    <span className="text-2xl font-bold font-mono text-slate-300 mt-1 block">
                      {result.news_risk_score}
                    </span>
                  </div>
                  <div className="bg-indigo-950/40 p-4 rounded-xl border border-indigo-900/60">
                    <span className="text-xs text-indigo-400 block font-bold">Final Risk</span>
                    <span className="text-3xl font-extrabold font-mono text-indigo-400 mt-1 block">
                      {result.multimodal_fusion?.final_multimodal_risk_score}
                    </span>
                  </div>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Risk Category:</span>
                    <span className="font-bold text-slate-200">
                      {result.multimodal_fusion?.risk_category}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">Financial Model Weight:</span>
                    <span className="font-mono text-indigo-400">
                      {((result.multimodal_fusion?.weights?.financial_weight || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">News Model Weight:</span>
                    <span className="font-mono text-slate-300">
                      {((result.multimodal_fusion?.weights?.news_weight || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-16 text-slate-500 text-sm">
                Fill in parameters on the left and click "Execute Multimodal Risk Prediction".
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
