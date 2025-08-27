import React, { useState } from 'react';
import { Button, Form, Spinner } from 'react-bootstrap';
import axios from 'axios';

function NoiseTest({ trainId, setLastPrediction }) {
  const [noiseLevel, setNoiseLevel] = useState(0.05);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const apiUrl = "http://localhost:8000/api";

  const handleTestNoise = async () => {
    if (!trainId) return;
    setLoading(true);
    setResult(null);
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.post(`${apiUrl}/predict_with_noise/`, {
        train_id: trainId,
        noise_level: noiseLevel
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(data);
      setLastPrediction(data);
    } catch (error) {
      console.error('Error al probar con ruido:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h6 className="fw-bold">Pruebas de Robustez (Ruido)</h6>
      <Form.Label>Nivel de ruido (0.01 - 0.5)</Form.Label>
      <Form.Range
        min={0.01}
        max={0.5}
        step={0.01}
        value={noiseLevel}
        onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
      />
      <div className="d-flex align-items-center mb-2">
        <div className="me-2">{noiseLevel.toFixed(2)}</div>
        <Button size="sm" variant="secondary" onClick={handleTestNoise} disabled={loading}>
          {loading ? <Spinner size="sm" animation="border" /> : 'Probar con Ruido'}
        </Button>
      </div>

      {result && (
        <div>
          <p className="mb-1"><strong>Predicción con ruido:</strong> {result.predicted_class}</p>
          <p className="mb-0"><strong>Tiempo de inferencia:</strong> {result.inference_time_ms} ms</p>
          <p className="mb-0"><strong>Tensor real:</strong> {result.tensor_file} </p>

        </div>
      )}
    </div>
  );
}

export default NoiseTest;
