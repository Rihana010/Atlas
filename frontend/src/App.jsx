import React, { useState } from 'react';
import { Layers, ChevronRight } from 'lucide-react';
import UploadStep from './components/UploadStep';
import DatasetStep from './components/DatasetStep';
import TrainingStep from './components/TrainingStep';
import ResultsStep from './components/ResultsStep';

export default function App() {
  const [step, setStep] = useState(1); // 1: Upload, 2: Dataset, 3: Training, 4: Results
  const [experimentId, setExperimentId] = useState(null);
  const [targetColumn, setTargetColumn] = useState(null);
  const [results, setResults] = useState(null);

  const handleUploadSuccess = (expId) => {
    setExperimentId(expId);
    setStep(2);
  };

  const handleTrainStarted = (targetCol) => {
    setTargetColumn(targetCol);
    setStep(3);
  };

  const handleTrainingFinished = (finalResults) => {
    setResults(finalResults);
    setStep(4);
  };

  const handleReset = () => {
    setStep(1);
    setExperimentId(null);
    setTargetColumn(null);
    setResults(null);
  };

  const steps = [
    { num: 1, label: 'Upload' },
    { num: 2, label: 'Dataset Analysis' },
    { num: 3, label: 'Model Training' },
    { num: 4, label: 'Results & Comparison' },
  ];

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <div className="brand-icon">A</div>
          <div>
            <div className="brand-title">ATLAS</div>
            <div className="brand-tagline">Automated ML Baseline Engine</div>
          </div>
        </div>

        {/* Linear Stepper */}
        <div className="stepper">
          {steps.map((s, idx) => {
            const isActive = step === s.num;
            const isCompleted = step > s.num;
            return (
              <React.Fragment key={s.num}>
                <div className={`step-item ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}>
                  <div className="step-num">{s.num}</div>
                  <span>{s.label}</span>
                </div>
                {idx < steps.length - 1 && <ChevronRight className="step-divider" size={14} />}
              </React.Fragment>
            );
          })}
        </div>
      </header>

      {/* Main Step Content */}
      <main>
        {step === 1 && <UploadStep onUploadSuccess={handleUploadSuccess} />}
        {step === 2 && experimentId && (
          <DatasetStep experimentId={experimentId} onTrainStarted={handleTrainStarted} />
        )}
        {step === 3 && experimentId && (
          <TrainingStep experimentId={experimentId} onTrainingFinished={handleTrainingFinished} />
        )}
        {step === 4 && experimentId && (
          <ResultsStep experimentId={experimentId} results={results} onReset={handleReset} />
        )}
      </main>
    </div>
  );
}
