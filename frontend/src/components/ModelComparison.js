import React, { useState, useEffect } from 'react';
import { Button, Form, Spinner, Alert } from 'react-bootstrap';
import axios from 'axios';

function ModelComparison({ currentTrainId }) {
  const [trainings, setTrainings] = useState([]);
  const [selectedTrainId, setSelectedTrainId] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const apiUrl = "http://localhost:8000/api";

  useEffect(() => {
    const fetchTrainings = async () => {
      try {
        const token = localStorage.getItem('access');
        const { data } = await axios.get(`${apiUrl}/list_user_trainings/`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setTrainings(data);
        if (data.length > 0) setSelectedTrainId(data[0].id);
      } catch (error) {
        console.error('Error al cargar entrenamientos:', error);
      }
    };
    fetchTrainings();
  }, []);

  const handleCompare = async () => {
    if (!selectedTrainId || !currentTrainId) return;
    setLoading(true);
    setResult(null);
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.post(`${apiUrl}/compare_models/`, {
        train_id_1: currentTrainId,
        train_id_2: selectedTrainId
      }, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setResult(data);
    } catch (error) {
      console.error('Error al comparar modelos:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h6 className="fw-bold">🔍 Comparación entre Modelos</h6>
      <Form.Group>
        <Form.Label>Selecciona otro entrenamiento para comparar</Form.Label>
        <Form.Select value={selectedTrainId} onChange={(e) => setSelectedTrainId(e.target.value)}>
          {trainings.filter(t => t.id !== currentTrainId).map(train => (
            <option key={train.id} value={train.id}>{train.training_name} ({train.model_type})</option>
          ))}
        </Form.Select>
      </Form.Group>
      <Button size="sm" className="mt-2" variant="primary" onClick={handleCompare} disabled={loading}>
        {loading ? <Spinner size="sm" animation="border" /> : 'Comparar Modelos'}
      </Button>
      {result && (
        <Alert variant="info" className="mt-2">
          <p className="mb-1"><strong>Modelo actual:</strong> {result.model_1.predicted_class} en {result.model_1.inference_time_ms} ms</p>
          <p className="mb-0"><strong>Modelo comparado:</strong> {result.model_2.predicted_class} en {result.model_2.inference_time_ms} ms</p>
        </Alert>
      )}
    </div>
  );
}

export default ModelComparison;
