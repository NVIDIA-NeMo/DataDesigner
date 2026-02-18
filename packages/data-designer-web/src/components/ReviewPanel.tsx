import { useState } from "react";
import {
  Sparkles,
  Loader2,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  Lightbulb,
} from "lucide-react";
import { api } from "../hooks/useApi";

interface Props {
  models: Record<string, unknown>[];
  loaded: boolean;
}

interface ReviewResult {
  static_issues: {
    level: string;
    type: string;
    column: string | null;
    message: string;
  }[];
  llm_tips: {
    category: string;
    severity: string;
    column: string | null;
    tip: string;
  }[];
  model_used: string;
}

export default function ReviewPanel({ models, loaded }: Props) {
  const [reviewModel, setReviewModel] = useState("");
  const [reviewing, setReviewing] = useState(false);
  const [result, setResult] = useState<ReviewResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleReview = async () => {
    if (!reviewModel) return;
    setReviewing(true);
    setResult(null);
    setError(null);
    try {
      const r = await api.reviewConfig(reviewModel);
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setReviewing(false);
    }
  };

  if (!loaded) {
    return (
      <div className="text-center py-12">
        <Sparkles size={32} className="text-gray-600 mx-auto mb-3" />
        <p className="text-gray-500">Load a config to run a review.</p>
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Controls */}
      <div className="card">
        <h2 className="text-sm font-semibold text-gray-300 mb-3">
          AI Config Review
        </h2>
        <p className="text-xs text-gray-500 mb-4">
          Run static analysis and get AI-powered suggestions to improve your
          config. Select a model to use for the review.
        </p>
        <div className="flex items-center gap-3">
          <select
            className="select-field flex-1"
            value={reviewModel}
            onChange={(e) => setReviewModel(e.target.value)}
          >
            <option value="">Select a model for review...</option>
            {models.map((m: any) => (
              <option key={m.alias} value={m.alias}>
                {m.alias} ({m.model})
              </option>
            ))}
          </select>
          <button
            className="btn-primary flex items-center gap-2"
            onClick={handleReview}
            disabled={reviewing || !reviewModel}
          >
            {reviewing ? (
              <Loader2 size={14} className="animate-spin" />
            ) : (
              <Sparkles size={14} />
            )}
            {reviewing ? "Reviewing..." : "Review"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card border-red-700/50 bg-red-900/20">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          {/* Static issues */}
          {result.static_issues.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-300 mb-3">
                Static Analysis ({result.static_issues.length})
              </h3>
              <div className="space-y-2">
                {result.static_issues.map((issue, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-3 text-sm rounded-lg px-4 py-3 ${
                      issue.level === "ERROR"
                        ? "bg-red-900/20 border border-red-700/30"
                        : "bg-amber-900/15 border border-amber-700/30"
                    }`}
                  >
                    {issue.level === "ERROR" ? (
                      <XCircle size={16} className="text-red-400 mt-0.5 shrink-0" />
                    ) : (
                      <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
                    )}
                    <div>
                      {issue.column && (
                        <span className="font-mono text-gray-400 mr-2 text-xs">
                          {issue.column}
                        </span>
                      )}
                      <span
                        className={
                          issue.level === "ERROR"
                            ? "text-red-300"
                            : "text-amber-300"
                        }
                      >
                        {issue.message}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* AI tips */}
          {result.llm_tips.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-300 mb-3">
                AI Suggestions ({result.llm_tips.length})
                <span className="text-xs text-gray-500 font-normal ml-2">
                  via {result.model_used}
                </span>
              </h3>
              <div className="space-y-2">
                {result.llm_tips.map((tip, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-3 text-sm bg-surface-2 border border-border rounded-lg px-4 py-3"
                  >
                    {tip.severity === "warning" ? (
                      <AlertTriangle size={16} className="text-amber-400 mt-0.5 shrink-0" />
                    ) : tip.severity === "suggestion" ? (
                      <Lightbulb size={16} className="text-purple-400 mt-0.5 shrink-0" />
                    ) : (
                      <Info size={16} className="text-blue-400 mt-0.5 shrink-0" />
                    )}
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] uppercase tracking-wider font-medium text-gray-500 bg-surface-3 px-2 py-0.5 rounded">
                          {tip.category.replace("_", " ")}
                        </span>
                        {tip.column && (
                          <span className="font-mono text-gray-500 text-xs">
                            {tip.column}
                          </span>
                        )}
                      </div>
                      <span className="text-gray-300">{tip.tip}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.static_issues.length === 0 &&
            result.llm_tips.length === 0 && (
              <div className="card border-green-700/30">
                <p className="text-sm text-green-400 flex items-center gap-2">
                  <CheckCircle2 size={16} />
                  Config looks good! No issues or suggestions.
                </p>
              </div>
            )}
        </div>
      )}
    </div>
  );
}
