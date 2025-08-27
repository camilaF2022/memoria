import React, { useState, useEffect } from 'react';
import { Modal, Button, Form, Alert } from 'react-bootstrap';
import axios from 'axios';

function GenerateModelModal({ isOpen, onClose, onCreated, modeloInicial }) {
  const [formData, setFormData] = useState({
    name: '',
    datasetId: '',
    modelType: 'CNN',
  });
  const [datasets, setDatasets] = useState([]);
  const [error, setError] = useState('');
  const apiUrl = "http://localhost:8000/api";

  useEffect(() => {
    if (isOpen) {
      fetchDatasets();
      setError('');

      if (modeloInicial) {
        setFormData({
          name: modeloInicial.name || '',
          datasetId: modeloInicial.dataset || modeloInicial.dataset_id || '',
          modelType: modeloInicial.model_type || 'CNN',
        });
      } else {
        setFormData({
          name: '',
          datasetId: '',
          modelType: 'CNN',
        });
      }
    }
  }, [isOpen, modeloInicial]);

  const fetchDatasets = async () => {
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.get(`${apiUrl}/datasets/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDatasets(data);
    } catch (error) {
      console.error('Error al obtener datasets:', error);
      setError('No se pudieron cargar los datasets.');
    }
  };

  const handleChange = ({ target: { name, value } }) =>
    setFormData(prev => ({ ...prev, [name]: value }));

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!formData.name.trim()) return setError('Ingresa un nombre para el modelo.');
    if (!formData.datasetId) return setError('Selecciona un dataset.');
    setError('');

    try {
      const payload = {
        name: formData.name,
        dataset: formData.datasetId,
        model_type: formData.modelType
      };

      const token = localStorage.getItem('access');
      if (!token) {
        setError('No estás autenticado. Inicia sesión.');
        return;
      }

      let response;

      if (modeloInicial && modeloInicial.id) {
        // Modo edición
        response = await axios.put(
          `${apiUrl}/models/${modeloInicial.id}/`,
          payload,
          { headers: { Authorization: `Bearer ${token}` } }
        );
      } else {
        // Modo creación
        response = await axios.post(
          `${apiUrl}/models/`,
          payload,
          { headers: { Authorization: `Bearer ${token}` } }
        );
      }

      onCreated(response.data);
      onClose();
    } catch (error) {
      console.error('Error al guardar modelo:', error);
      setError('Error al guardar el modelo. Revisa los datos.');
    }
  };

  return (
    <Modal show={isOpen} onHide={onClose} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>{modeloInicial ? 'Editar Modelo' : 'Generar Modelo de Entrenamiento'}</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}

        <Form onSubmit={handleSubmit}>
          <Form.Group>
            <Form.Label>Nombre del Modelo</Form.Label>
            <Form.Control
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="Ej: Modelo CNN sensores"
              required
            />
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Seleccionar Dataset</Form.Label>
            <Form.Select
              name="datasetId"
              value={formData.datasetId}
              onChange={handleChange}
              required
            >
              <option value="">-- Selecciona un dataset --</option>
              {datasets.map(ds => (
                <option key={ds.id} value={ds.id}>
                  {ds.name} ({ds.data_type})
                </option>
              ))}
            </Form.Select>
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Tipo de Modelo</Form.Label>
            <Form.Select
              name="modelType"
              value={formData.modelType}
              onChange={handleChange}
              required
            >
              <option value="CNN">CNN</option>
              <option value="LSTM">LSTM</option>
              <option value="NAIVE_BAYES">Naive Bayes</option>
              <option value="KMEANS">KMeans</option> 
              <option value="SVM">SVM</option> 
              <option value="YOLO">yolo</option>


              </Form.Select>
          </Form.Group>

          <Button className="mt-3" type="submit">
            {modeloInicial ? 'Guardar Cambios' : 'Crear Modelo'}
          </Button>
        </Form>
      </Modal.Body>
    </Modal>
  );
}

export default GenerateModelModal;
