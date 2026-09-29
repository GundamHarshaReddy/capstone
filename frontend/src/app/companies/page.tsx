"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchCompanies } from "@/lib/api";
import { Search, ChevronLeft, ChevronRight, Filter } from "lucide-react";

export default function CompaniesPage() {
  const [companies, setCompanies] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const res = await fetchCompanies(page, 20, search);
        setCompanies(res.companies || []);
        setTotal(res.total || 0);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [page, search]);

  const totalPages = Math.ceil(total / 20);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Company Directory</h1>
          <p className="text-slate-400 text-sm">Browse financial distress and news risk scores across 422 companies.</p>
        </div>

        {/* Search Bar */}
        <div className="relative w-full sm:w-72">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
          <input
            type="text"
            placeholder="Search Company ID..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg pl-9 pr-4 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* Companies Grid/Table */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-sm">
        {loading ? (
          <div className="p-12 text-center text-slate-400 text-sm">Loading company records...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="bg-slate-950/60 border-b border-slate-800 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                <tr>
                  <th className="py-3.5 px-4">Company ID</th>
                  <th className="py-3.5 px-4">Latest Period</th>
                  <th className="py-3.5 px-4">Historical Target</th>
                  <th className="py-3.5 px-4">Financial Risk Score</th>
                  <th className="py-3.5 px-4">News Risk Score</th>
                  <th className="py-3.5 px-4">Multimodal Risk</th>
                  <th className="py-3.5 px-4">Risk Category</th>
                  <th className="py-3.5 px-4 text-right">Details</th>
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
                      <td className="py-3 px-4 font-semibold text-slate-200">Company {comp.company_id}</td>
                      <td className="py-3 px-4 text-slate-400">Time {comp.latest_time_period}</td>
                      <td className="py-3 px-4">
                        {comp.is_distressed_historical === 1 ? (
                          <span className="text-xs bg-red-950 text-red-300 px-2 py-0.5 rounded font-medium">Distressed</span>
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
        )}

        {/* Pagination Footer */}
        <div className="p-4 bg-slate-950/40 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <div>
            Showing <span className="text-slate-200 font-medium">{companies.length}</span> of{" "}
            <span className="text-slate-200 font-medium">{total}</span> companies
          </div>
          <div className="flex items-center space-x-2">
            <button
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="p-1.5 rounded bg-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-700"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <span>
              Page {page} of {totalPages || 1}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage(page + 1)}
              className="p-1.5 rounded bg-slate-800 text-slate-300 disabled:opacity-40 hover:bg-slate-700"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
