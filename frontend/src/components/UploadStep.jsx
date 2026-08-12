import React, { useState } from 'react';
import { Upload, AlertCircle, FileText } from 'lucide-react';

export default function UploadStep({ onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      const lowerName = selected.name.toLowerCase();
      
      if (!lowerName.endsWith('.csv') && !lowerName.endsWith('.txt') && !lowerName.endsWith('.zip')) {
        setError('Please select a valid CSV (.csv) or ZIP (.zip) dataset archive.');
        return;
      }
      setFile(selected);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed.');
      }

      onUploadSuccess(data.experiment_id);
    } catch (err) {
      setError(err.message || 'An error occurred during dataset upload.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">
          <Upload size={16} color="var(--status-blue)" />
          Dataset Upload
        </span>
        <span className="badge badge-blue">CSV / ZIP DATASETS</span>
      </div>

      <div className="dropzone" onClick={() => document.getElementById('csv-input').click()}>
        <input
          type="file"
          id="csv-input"
          accept=".csv, .zip, .CSV, .ZIP, text/csv, application/zip, application/x-zip-compressed, text/plain, */*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <FileText size={36} color="var(--text-secondary)" style={{ marginBottom: 12 }} />
        <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
          {file ? file.name : 'Select or drop a CSV or ZIP dataset file'}
        </h4>
        <p style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Supports CSV files & Kaggle ZIP dataset archives'}
        </p>
      </div>

      {error && (
        <div style={{ marginTop: 16, padding: 12, background: 'var(--status-red-bg)', border: '1px solid var(--status-red)', borderRadius: 3, display: 'flex', alignItems: 'center', gap: 8, color: 'var(--status-red)' }}>
          <AlertCircle size={16} />
          <span style={{ fontSize: 13 }}>{error}</span>
        </div>
      )}

      <div style={{ marginTop: 20, display: 'flex', justifyContent: 'flex-end' }}>
        <button
          className="btn btn-primary"
          onClick={handleUpload}
          disabled={!file || isUploading}
        >
          {isUploading ? (
            <>
              <div className="spinner" />
              <span>Uploading & Unpacking...</span>
            </>
          ) : (
            'Proceed to Dataset Analysis'
          )}
        </button>
      </div>
    </div>
  );
}
