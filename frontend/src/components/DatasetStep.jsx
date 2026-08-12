import React, { useEffect, useState } from 'react';
import { Database, Play, AlertCircle, Table } from 'lucide-react';

export default function DatasetStep({ experimentId, onTrainStarted }) {
  const [dataset, setDataset] = useState(null);
  const [loading, setLoading] = useState(true);
  const [targetColumn, setTargetColumn] = useState('');
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function fetchDataset() {
      try {
        const response = await fetch(`/api/dataset/${experimentId}`);
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || 'Failed to load dataset details.');
        }
        setDataset(data);
        if (data.columns && data.columns.length > 0) {
          // Preselect last column as default target candidate
          setTargetColumn(data.columns[data.columns.length - 1].name);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    fetchDataset();
  }, [experimentId]);

  const handleStartTraining = async () => {
    if (!targetColumn) {
      setError('Please select a target column before starting training.');
      return;
    }

    setIsStarting(true);
    setError(null);

    try {
      const response = await fetch('/api/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          experiment_id: experimentId,
          target_column: targetColumn,
        }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to initiate training.');
      }

      onTrainStarted(targetColumn);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsStarting(false);
    }
  };

  if (loading) {
    return (
      <div className="panel" style={{ textAlign: 'center', padding: 48 }}>
        <div className="spinner" style={{ margin: '0 auto 16px auto' }} />
        <span style={{ color: 'var(--text-secondary)' }}>Analyzing dataset structure and column statistics...</span>
      </div>
    );
  }

  if (error && !dataset) {
    return (
      <div className="panel">
        <div style={{ padding: 16, background: 'var(--status-red-bg)', border: '1px solid var(--status-red)', borderRadius: 3, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--status-red)' }}>
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div>
      {/* Overview & Target Selection */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">
            <Database size={16} color="var(--status-blue)" />
            Dataset Analysis — <span className="mono">{dataset.dataset_name}</span>
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <span className="badge badge-blue">{dataset.total_rows.toLocaleString()} ROWS</span>
            <span className="badge badge-blue">{dataset.total_columns} COLUMNS</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 20, alignItems: 'end', marginBottom: 20 }}>
          <div>
            <label style={{ display: 'block', fontSize: 12, color: 'var(--text-secondary)', marginBottom: 6, fontWeight: 600 }}>
              SELECT TARGET COLUMN TO PREDICT
            </label>
            <select
              className="select-input"
              style={{ width: '100%' }}
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
            >
              {dataset.columns.map((col) => (
                <option key={col.name} value={col.name}>
                  {col.name} ({col.dtype} — {col.unique_count} unique)
                </option>
              ))}
            </select>
          </div>

          <div>
            <button
              className="btn btn-primary"
              style={{ width: '100%', height: 38 }}
              onClick={handleStartTraining}
              disabled={isStarting || !targetColumn}
            >
              {isStarting ? (
                <>
                  <div className="spinner" />
                  <span>Queuing Async Training...</span>
                </>
              ) : (
                <>
                  <Play size={14} />
                  <span>Start Model Training</span>
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div style={{ padding: 12, background: 'var(--status-red-bg)', border: '1px solid var(--status-red)', borderRadius: 3, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--status-red)', marginBottom: 16 }}>
            <AlertCircle size={16} />
            <span style={{ fontSize: 13 }}>{error}</span>
          </div>
        )}

        {/* Column Stats Table */}
        <h4 style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 10, letterSpacing: '0.04em' }}>
          SCHEMA & COLUMN METRICS
        </h4>
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Column Name</th>
                <th>Data Type</th>
                <th>Null Count</th>
                <th>Null %</th>
                <th>Unique Values</th>
                <th>Sample Values</th>
              </tr>
            </thead>
            <tbody>
              {dataset.columns.map((col) => (
                <tr key={col.name} style={{ background: col.name === targetColumn ? 'rgba(88, 166, 255, 0.05)' : undefined }}>
                  <td style={{ fontWeight: col.name === targetColumn ? 'bold' : 'normal' }}>
                    {col.name} {col.name === targetColumn && <span className="badge badge-blue" style={{ marginLeft: 6 }}>TARGET</span>}
                  </td>
                  <td>{col.dtype}</td>
                  <td>{col.null_count.toLocaleString()}</td>
                  <td>{col.null_percentage.toFixed(1)}%</td>
                  <td>{col.unique_count.toLocaleString()}</td>
                  <td style={{ color: 'var(--text-secondary)' }}>
                    {col.sample_values.slice(0, 3).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Preview */}
      <div className="panel">
        <div className="panel-header">
          <span className="panel-title">
            <Table size={16} color="var(--text-secondary)" />
            Dataset Preview (First 10 Rows)
          </span>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                {dataset.columns.map((col) => (
                  <th key={col.name}>{col.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {dataset.preview.map((row, idx) => (
                <tr key={idx}>
                  {dataset.columns.map((col) => (
                    <td key={col.name}>{row[col.name] !== undefined ? String(row[col.name]) : ''}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
