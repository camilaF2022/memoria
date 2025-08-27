import React, { useState, useEffect } from 'react';
import { Button, Form, Spinner, Alert } from 'react-bootstrap';
import axios from 'axios';

function LabelTester({ trainId, setLastPrediction }) {
  const [labels, setLabels] = useState([]);
  const [selectedLabel, setSelectedLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const apiUrl = "http://localhost:8000/api";

  useEffect(() => {
    const fetchLabels = async () => {
      if (!trainId) return;
      try {
        const token = localStorage.getItem('access');
        const { data } = await axios.get(`${apiUrl}/get_labels/?train_id=${trainId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setLabels(data.labels);
        if (data.labels.length > 0) setSelectedLabel(data.labels[0]);
      } catch (error) {
        console.error('Error al cargar labels:', error);
      }
    };
    fetchLabels();
  }, [trainId]);

  const handleTestLabel = async () => {
    if (!trainId || !selectedLabel) return;
    setLoading(true);
    setResult(null);
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.post(`${apiUrl}/predict_random_tensor/`, {
        train_id: trainId,
        label: selectedLabel
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(data);
      setLastPrediction(data);
    } catch (error) {
      console.error('Error al probar con label:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h6 className="fw-bold">Probar Predicción por Label</h6>
      <Form.Group>
        <Form.Label>Selecciona un label para probar</Form.Label>
        <Form.Select value={selectedLabel} onChange={(e) => setSelectedLabel(e.target.value)}>
          {labels.map(label => (
            <option key={label} value={label}>{label}</option>
          ))}
        </Form.Select>
      </Form.Group>
      <Button size="sm" className="mt-2" variant="secondary" onClick={handleTestLabel} disabled={loading}>
        {loading ? <Spinner size="sm" animation="border" /> : 'Probar Label'}
      </Button>
      {result && (
        <Alert variant="info" className="mt-2">
          <p className="mb-1"><strong>Clase Predicha:</strong> {result.predicted_class}</p>
          <p className="mb-0"><strong>Tensor utilizado:</strong> {result.tensor_file}</p>
        </Alert>
      )}
    </div>
  );
}

export default LabelTester;