import React from 'react';
import { Award, AlertCircle, RefreshCw, BarChart2, CheckCircle2, Clock } from 'lucide-react';

export default function ResultsStep({ experimentId, results, onReset }) {
  if (!results) {
    return (
      <div className="panel" style={{ textAlign: 'center', padding: 48 }}>
        <span style={{ color: 'var(--text-secondary)' }}>No results found for experiment {experimentId}.</span>
      </div>
    );
  }

  if (results.status === 'failed') {
    return (
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title" style={{ color: 'var(--status-red)' }}>
            <AlertCircle size={18} />
            Training Execution Failed
          </span>
          <span className="badge badge-red">FAILED</span>
        </div>

        <div style={{ padding: 16, background: 'var(--status-red-bg)', border: '1px solid var(--status-red)', borderRadius: 4, marginBottom: 20 }}>
          <h4 style={{ color: 'var(--status-red)', fontWeight: 600, marginBottom: 6 }}>Error Details:</h4>
          <pre className="mono" style={{ whiteSpace: 'pre-wrap', fontSize: 13, color: 'var(--text-primary)' }}>
            {results.error_message || 'An unknown error occurred during preprocessing or model fitting.'}
          </pre>
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={onReset}>
            <RefreshCw size={14} />
            Start New Experiment
          </button>
        </div>
      </div>
    );
  }

  const isClassification = results.problem_type === 'classification';

  return (
    <div>
      {/* Overview & Best Model Banner */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">
            <BarChart2 size={18} color="var(--status-blue)" />
            Model Evaluation & Baseline Comparison
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-blue mono">{results.experiment_id}</span>
            <span className="badge badge-green">{results.status.toUpperCase()}</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16, marginBottom: 20 }}>
          <div style={{ background: 'var(--bg-dark)', border: '1px solid var(--panel-border)', padding: 14, borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 600 }}>TARGET COLUMN</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 4 }} className="mono">
              {results.target_column}
            </div>
          </div>

          <div style={{ background: 'var(--bg-dark)', border: '1px solid var(--panel-border)', padding: 14, borderRadius: 4 }}>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', fontWeight: 600 }}>PROBLEM TYPE</div>
            <div style={{ fontSize: 15, fontWeight: 700, marginTop: 4, color: 'var(--status-blue)' }} className="mono">
              {results.problem_type ? results.problem_type.toUpperCase() : 'N/A'}
            </div>
          </div>

          <div style={{ background: 'var(--bg-dark)', border: '1px solid var(--status-green)', padding: 14, borderRadius: 4, gridColumn: 'span 2' }}>
            <div style={{ fontSize: 11, color: 'var(--status-green)', fontWeight: 600, display: 'flex', alignItems: 'center', gap: 6 }}>
              <Award size={14} />
              BEST PERFORMING MODEL
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, marginTop: 4, color: 'var(--text-primary)' }} className="mono">
              {results.best_model || 'N/A'}
            </div>
          </div>
        </div>

        {results.reasoning && (
          <div className="reasoning-box" style={{ marginBottom: 20 }}>
            <strong style={{ display: 'block', marginBottom: 2, color: 'var(--status-blue)' }}>Deterministic Reasoning:</strong>
            {results.reasoning}
          </div>
        )}
      </div>

      {/* Model Comparison Table */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">
            Performance Metrics Matrix
          </span>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Name</th>
                <th>Status</th>
                {isClassification ? (
                  <>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                  </>
                ) : (
                  <>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>R² Score</th>
                  </>
                )}
                <th>Training Time</th>
              </tr>
            </thead>
            <tbody>
              {results.metrics.map((m) => {
                const isBest = m.model_name === results.best_model;
                return (
                  <tr
                    key={m.model_name}
                    style={{
                      background: isBest ? 'rgba(63, 185, 80, 0.08)' : undefined,
                    }}
                  >
                    <td style={{ fontWeight: isBest ? 'bold' : 'normal' }}>
                      {m.model_name}{' '}
                      {isBest && <span className="badge badge-green" style={{ marginLeft: 6 }}>BEST MODEL</span>}
                    </td>
                    <td>
                      {m.model_status === 'completed' ? (
                        <span style={{ color: 'var(--status-green)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          <CheckCircle2 size={13} /> Completed
                        </span>
                      ) : (
                        <span style={{ color: 'var(--status-red)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                          <AlertCircle size={13} /> Failed
                        </span>
                      )}
                    </td>
                    {isClassification ? (
                      <>
                        <td>{m.accuracy !== null ? m.accuracy.toFixed(4) : '-'}</td>
                        <td>{m.precision !== null ? m.precision.toFixed(4) : '-'}</td>
                        <td>{m.recall !== null ? m.recall.toFixed(4) : '-'}</td>
                        <td style={{ fontWeight: 'bold', color: isBest ? 'var(--status-green)' : 'var(--text-primary)' }}>
                          {m.f1 !== null ? m.f1.toFixed(4) : '-'}
                        </td>
                      </>
                    ) : (
                      <>
                        <td>{m.rmse !== null ? m.rmse.toFixed(4) : '-'}</td>
                        <td>{m.mae !== null ? m.mae.toFixed(4) : '-'}</td>
                        <td style={{ fontWeight: 'bold', color: isBest ? 'var(--status-green)' : 'var(--text-primary)' }}>
                          {m.r2 !== null ? m.r2.toFixed(4) : '-'}
                        </td>
                      </>
                    )}
                    <td style={{ color: 'var(--text-secondary)' }}>
                      <Clock size={12} style={{ display: 'inline', marginRight: 4 }} />
                      {m.training_time !== null ? `${m.training_time.toFixed(3)}s` : '-'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={onReset}>
            <RefreshCw size={14} />
            Start New Experiment
          </button>
        </div>
      </div>
    </div>
  );
}
