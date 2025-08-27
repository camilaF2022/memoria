import React, { useState, useEffect } from 'react';
import { Modal, Button, Form, Alert } from 'react-bootstrap';
import axios from 'axios';             

function GenerateDataModal({ isOpen, onClose, onCreated, setLoading }) {
  const [formData, setFormData] = useState({
    name: '',
    dataType: '',
    sampleCount: 1000,
    outputFormat: 'CSV',
    description: '',
  });
  const [error, setError] = useState('');
  const apiUrl = "http://localhost:8000/api";

  useEffect(() => {
    if (isOpen) {
      setFormData({
        name: '',
        dataType: '',
        sampleCount: 1000,
        outputFormat: 'CSV',
        description: '',
      });
      setError('');
    }
  }, [isOpen]);

  const handleChange = ({ target: { name, value } }) =>
  setFormData(prev => ({ ...prev, [name]: value }));        

  const handleSubmit = async (e) => {
    e.preventDefault();
  
    if (!formData.dataType) return setError('Selecciona un tipo de datos.');
    if (!formData.name.trim()) return setError('Ingresa un nombre para el dataset.');
    setError('');
  
    try {
      const payload = {
        name: formData.name,
        description: formData.description,
        data_type: formData.dataType.toLowerCase(),
        sample_count: parseInt(formData.sampleCount, 10),
      };
  
      const token = localStorage.getItem('access'); 
  
      if (!token) {
        setError('No estás autenticado. Por favor, inicia sesión.');
        return;
      }
  
      setLoading(true);       // <-- activa el spinner
      onClose();              // <-- cierra el modal
  
      const { data } = await axios.post(
        `${apiUrl}/datasets/`,
        payload,
        {
          headers: { Authorization: `Bearer ${token}` }
        }
      );
  
      onCreated(data);
    } catch (error) {
      console.error('Error creando dataset:', error);
      setLoading(false);      // <-- si falla, desactiva el loading
    }
  };
  
  
  return (
    <Modal show={isOpen} onHide={onClose} size="lg" centered>
      <Modal.Header closeButton>
        <Modal.Title>Generar nuevo dataset</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}

        <Form onSubmit={handleSubmit}>
          <Form.Group>
            <Form.Label>Nombre del dataset</Form.Label>
            <Form.Control
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="Ej: Dataset de sensores"
              required
            />
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Tipo de datos</Form.Label>
            <Form.Select
              name="dataType"
              value={formData.dataType}
              onChange={handleChange}
              required
            >
              <option value="" disabled>-- Seleccione tipo de datos --</option>
              <option value="Sensor">Sensor</option>
              <option value="Audio">Audio</option>
              <option value="Video">Video</option>
            </Form.Select>
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Cantidad de muestras</Form.Label>
            <Form.Control
              type="number"
              name="sampleCount"
              value={formData.sampleCount}
              onChange={handleChange}
              min={1}
            />
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Descripción (opcional)</Form.Label>
            <Form.Control
              as="textarea"
              rows={3}
              name="description"
              value={formData.description}
              onChange={handleChange}
            />
          </Form.Group>

          <Button className="mt-3 btn-secondary" type="submit">Generar datos</Button>
        </Form>
      </Modal.Body>
    </Modal>
  );
}

export default GenerateDataModal;
