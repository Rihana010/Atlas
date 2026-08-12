import React, { useEffect, useState } from 'react';
import { Cpu, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function TrainingStep({ experimentId, onTrainingFinished }) {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let isMounted = true;
    let timerId = null;

    async function pollResults() {
      try {
        const response = await fetch(`/api/results/${experimentId}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to check experiment status.');
        }

        if (isMounted) {
          setResults(data);

          if (data.status === 'completed' || data.status === 'failed') {
            // Auto transition after short pause for smooth UX
            setTimeout(() => {
              if (isMounted) onTrainingFinished(data);
            }, 800);
          } else {
            // Poll again after 1.5s
            timerId = setTimeout(pollResults, 1500);
          }
        }
      } catch (err) {
        if (isMounted) setError(err.message);
      }
    }

    pollResults();

    return () => {
      isMounted = false;
      if (timerId) clearTimeout(timerId);
    };
  }, [experimentId, onTrainingFinished]);

  return (
    <div className="panel" style={{ textAlign: 'center', padding: '40px 24px' }}>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
        {results?.status === 'failed' ? (
          <AlertCircle size={48} color="var(--status-red)" />
        ) : results?.status === 'completed' ? (
          <CheckCircle2 size={48} color="var(--status-green)" />
        ) : (
          <div className="spinner" style={{ width: 44, height: 44, borderWidth: 3 }} />
        )}
      </div>

      <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>
        {results?.status === 'queued'
          ? 'Queued in Worker Pool...'
          : results?.status === 'running'
          ? 'Asynchronous Model Training In Progress'
          : results?.status === 'completed'
          ? 'Training Pipeline Completed'
          : results?.status === 'failed'
          ? 'Pipeline Training Failed'
          : 'Initializing Training Worker...'}
      </h3>

      <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginBottom: 24 }} className="mono">
        Experiment ID: {experimentId}
      </p>

      {results?.problem_type && (
        <div style={{ maxWidth: 640, margin: '0 auto', textAlign: 'left' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
              DETERMINISTIC PROBLEM TYPE:
            </span>
            <span className={`badge ${results.problem_type === 'classification' ? 'badge-blue' : 'badge-green'}`}>
              {results.problem_type.toUpperCase()}
            </span>
          </div>

          {results.reasoning && (
            <div className="reasoning-box">
              <strong style={{ display: 'block', marginBottom: 4, color: 'var(--status-blue)' }}>
                Detection Reasoning:
              </strong>
              {results.reasoning}
            </div>
          )}
        </div>
      )}

      {error && (
        <div style={{ maxWidth: 640, margin: '20px auto 0 auto', padding: 12, background: 'var(--status-red-bg)', border: '1px solid var(--status-red)', borderRadius: 3, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--status-red)' }}>
          <AlertCircle size={16} />
          <span style={{ fontSize: 13 }}>{error}</span>
        </div>
      )}
    </div>
  );
}
