import React, { useState, useEffect } from 'react';
import { Modal, Button, Form, Alert, Spinner } from 'react-bootstrap';
import axios from 'axios';

function TrainModelModal({ isOpen, onClose, onTrained, onStartTraining }) {
  const [formData, setFormData] = useState({
    trainingName: '',
    modelName: '',
    modeloId: '',
    modelType: '',
    epochs: 30,
    batchSize: 16,
    imgsz: 640,
    lr0: 0.01,
  });

  const [modelosDisponibles, setModelosDisponibles] = useState([]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const apiUrl = "http://localhost:8000";

  const [nbParams, setNbParams] = useState({
    var_smoothing: 1e-9,
    test_split: 0.2,
  });

  const [kmeansParams, setKmeansParams] = useState({
    n_clusters: 3,
  });

  const [svmParams, setSvmParams] = useState({
    kernel: 'rbf',
    C: 1.0,
    test_split: 0.2,
  });

  useEffect(() => {
    if (isOpen) {
      setFormData({
        trainingName: '',
        modelName: '',
        modeloId: '',
        modelType: '',
        epochs: 30,
        batchSize: 16,
        imgsz: 640,
        lr0: 0.01,
      });
      setError('');
      setLoading(false);
      fetchModelos();
    }
  }, [isOpen]);

  const fetchModelos = async () => {
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.get(`${apiUrl}/api/models/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setModelosDisponibles(data);
    } catch (error) {
      console.error('Error al cargar modelos:', error);
      setError('No se pudieron cargar los modelos disponibles.');
    }
  };

  const handleChange = ({ target: { name, value } }) => {
    if (name === 'modelName') {
      const modeloSeleccionado = modelosDisponibles.find(m => m.name === value);
      if (modeloSeleccionado) {
        setFormData(prev => ({
          ...prev,
          modelName: modeloSeleccionado.name,
          modeloId: modeloSeleccionado.id,
          modelType: modeloSeleccionado.model_type?.toLowerCase() || '',
        }));
      } else {
        setFormData(prev => ({
          ...prev,
          modelName: value,
          modeloId: '',
          modelType: ''
        }));
      }
    } else if (name === 'n_clusters') {
      setKmeansParams({ ...kmeansParams, n_clusters: parseInt(value, 10) });
    } else if (name === 'var_smoothing' || name === 'test_split') {
      setNbParams({ ...nbParams, [name]: parseFloat(value) });
    } else if (['kernel', 'C'].includes(name)) {
      setSvmParams(prev => ({
        ...prev,
        [name]: name === 'C' ? parseFloat(value) : value
      }));
    } else if (name === 'svm_test_split') {
      setSvmParams(prev => ({
        ...prev,
        test_split: parseFloat(value)
      }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.trainingName.trim()) return setError('Ingresa un nombre para el entrenamiento.');
    if (!formData.modelName.trim()) return setError('Selecciona un modelo.');
    if (!formData.modelType.trim()) return setError('El tipo de modelo no puede estar vacío.');
    setError('');
    setLoading(true);

    onStartTraining();
    onClose();

    try {
      const payload = {
        training_name: formData.trainingName,
        modelo_id: formData.modeloId,
        epochs: parseInt(formData.epochs, 10),
        batch_size: parseInt(formData.batchSize, 10),
      };

      if (formData.modelType === 'naive_bayes') {
        payload.params = {
          ...nbParams,
          test_split: parseFloat(nbParams.test_split),
          var_smoothing: parseFloat(nbParams.var_smoothing)
        };
        delete payload.epochs;
        delete payload.batch_size;
      } else if (formData.modelType === 'kmeans') {
        payload.params = {
          n_clusters: parseInt(kmeansParams.n_clusters, 10)
        };
        delete payload.epochs;
        delete payload.batch_size;
      } else if (formData.modelType === 'svm') {
        payload.params = {
          kernel: svmParams.kernel,
          C: parseFloat(svmParams.C),
          test_split: parseFloat(svmParams.test_split),
        };
        delete payload.epochs;
        delete payload.batch_size;
      } else if (formData.modelType === 'yolo') {
        payload.params = {
          imgsz: parseInt(formData.imgsz, 10),
          lr0: parseFloat(formData.lr0),
        };
      }

      const token = localStorage.getItem('access');
      const { data } = await axios.post(
        `${apiUrl}/api/trains/`,
        payload,
        { headers: { Authorization: `Bearer ${token}` } }
      );

      onTrained(data);
    } catch (error) {
      console.error('Error al entrenar:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal show={isOpen} onHide={onClose} centered size="lg">
      <Modal.Header closeButton>
        <Modal.Title>Entrenar modelo existente</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}

        <Form onSubmit={handleSubmit}>
          <Form.Group>
            <Form.Label>Nombre del entrenamiento</Form.Label>
            <Form.Control
              name="trainingName"
              value={formData.trainingName}
              onChange={handleChange}
              placeholder="Ej: Entrenamiento_LSTM_Sensores"
              required
            />
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Seleccionar modelo existente</Form.Label>
            <Form.Select
              name="modelName"
              value={formData.modelName}
              onChange={handleChange}
              required
            >
              <option value="">-- Selecciona un modelo --</option>
              {modelosDisponibles.map(modelo => (
                <option key={modelo.id} value={modelo.name}>
                  {modelo.name} ({modelo.model_type})
                </option>
              ))}
            </Form.Select>
          </Form.Group>

          <Form.Group className="mt-2">
            <Form.Label>Tipo de modelo (autocompletado)</Form.Label>
            <Form.Control
              name="modelType"
              value={formData.modelType}
              disabled
              placeholder="Tipo de modelo"
            />
          </Form.Group>

          {formData.modelType === 'yolo' && (
            <>
            <Form.Group className="mt-2">
                <Form.Label>Épocas</Form.Label>
                <Form.Control
                  type="number"
                  name="epochs"
                  value={formData.epochs}
                  onChange={handleChange}
                  min={1}
                />
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Resolución de entrada (imgsz)</Form.Label>
                <Form.Control
                  type="number"
                  name="imgsz"
                  value={formData.imgsz}
                  onChange={handleChange}
                  min={64}
                />
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Learning Rate inicial (lr0)</Form.Label>
                <Form.Control
                  type="number"
                  name="lr0"
                  value={formData.lr0}
                  onChange={handleChange}
                  step="0.0001"
                  min={0}
                />
              </Form.Group>
            </>
          )}

          {formData.modelType === 'kmeans' ? (
            <Form.Group className="mt-2">
              <Form.Label>Número de clusters (k)</Form.Label>
              <Form.Control
                type="number"
                name="n_clusters"
                value={kmeansParams.n_clusters}
                min={1}
                step={1}
                onChange={(e) =>
                  setKmeansParams({ ...kmeansParams, n_clusters: e.target.value })
                }
              />
            </Form.Group>
          ) : formData.modelType === 'naive_bayes' ? (
            <>
              <Form.Group className="mt-2">
                <Form.Label>Var Smoothing</Form.Label>
                <Form.Control
                  type="number"
                  step="1e-9"
                  name="var_smoothing"
                  value={nbParams.var_smoothing}
                  onChange={handleChange}
                />
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Test Split (%)</Form.Label>
                <Form.Control
                  type="number"
                  step="0.01"
                  name="test_split"
                  value={nbParams.test_split}
                  onChange={handleChange}
                />
              </Form.Group>
            </>
          ) : formData.modelType === 'svm' ? (
            <>
              <Form.Group className="mt-2">
                <Form.Label>Kernel</Form.Label>
                <Form.Select
                  name="kernel"
                  value={svmParams.kernel}
                  onChange={handleChange}
                >
                  <option value="rbf">RBF (default)</option>
                  <option value="linear">Linear</option>
                  <option value="poly">Polynomial</option>
                  <option value="sigmoid">Sigmoid</option>
                  <option value="yolo">Yolo</option>
                </Form.Select>
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Parámetro C</Form.Label>
                <Form.Control
                  type="number"
                  step="0.1"
                  name="C"
                  value={svmParams.C}
                  onChange={handleChange}
                />
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Test Split (%)</Form.Label>
                <Form.Control
                  type="number"
                  step="0.01"
                  name="svm_test_split"
                  value={svmParams.test_split}
                  onChange={handleChange}
                />
              </Form.Group>
            </>
          ) : formData.modelType !== 'yolo' && (
            <>
              <Form.Group className="mt-2">
                <Form.Label>Épocas</Form.Label>
                <Form.Control
                  type="number"
                  name="epochs"
                  value={formData.epochs}
                  onChange={handleChange}
                  min={1}
                />
              </Form.Group>
              <Form.Group className="mt-2">
                <Form.Label>Tamaño del batch</Form.Label>
                <Form.Control
                  type="number"
                  name="batchSize"
                  value={formData.batchSize}
                  onChange={handleChange}
                  min={1}
                />
              </Form.Group>
            </>
          )}

          <Button className="mt-3" type="submit" disabled={loading}>
            {loading ? (
              <>
                <Spinner animation="border" size="sm" /> Entrenando...
              </>
            ) : (
              'Iniciar entrenamiento'
            )}
          </Button>
        </Form>
      </Modal.Body>
    </Modal>
  );
}

export default TrainModelModal;
