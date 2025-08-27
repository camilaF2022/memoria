import React, { useState, useEffect } from 'react';
import GenerateDataModal from '../components/GenerateDataModal';
import VisualizeDataModal from '../components/VisualizeDataModal';
import axios from 'axios';
import { Modal, Button, Form } from 'react-bootstrap';

function Datos() {
  const [showModal, setShowModal] = useState(false);
  const [datasets, setDatasets] = useState([]);
  const [showVisualizeModal, setShowVisualizeModal] = useState(false);
  const [selectedDatasetId, setSelectedDatasetId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [infoType, setInfoType] = useState(null);
  const apiUrl = "http://localhost:8000/api";

  const fetchDatasets = async () => {
    try {
      const response = await axios.get(`${apiUrl}/datasets/`, {
        headers: {
          Authorization: `Bearer ${localStorage.getItem('access')}`,
        },
      });
      setDatasets(response.data);
    } catch (error) {
      console.error('Error cargando datasets', error);
    }
  };

  const handleCsvUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      setIsLoading(true);
      const response = await axios.post(`${apiUrl}/datasets/upload/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${localStorage.getItem('access')}`,
        },
      });
      if (response.status === 201) {
        alert('Archivo subido correctamente');
        fetchDatasets();
      }
    } catch (error) {
      console.error('Error al subir CSV:', error);
      alert('Error al subir el archivo CSV');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      const hasInProgress = datasets.some(
        d => d.status === 'pendiente' || d.status === 'en_progreso'
      );
      if (hasInProgress) fetchDatasets();
    }, 4000);
    return () => clearInterval(interval);
  }, [datasets]);

  const handleDatasetCreated = (nuevoDataset) => {
    setDatasets(prev => [...prev, nuevoDataset]);
    setIsLoading(false);
  };

  const renderStatusBadge = (status) => {
    const normalized = status?.toLowerCase();
    const label = normalized?.replace('_', ' ');
    switch (normalized) {
      case 'pendiente': return <span className="badge bg-secondary">{label}</span>;
      case 'en_progreso': return <span className="badge bg-warning text-dark">{label}</span>;
      case 'completado': return <span className="badge bg-success">{label}</span>;
      case 'fallido': return <span className="badge bg-danger">{label}</span>;
      default: return <span className="badge bg-light text-dark">Desconocido</span>;
    }
  };

  const renderInfoModal = () => {
    let content = null;
    if (infoType === 'sensor') {
      content = (
        <>
          <p>Los datos de sensores provienen de <strong>acelerómetros</strong>, <strong>giroscopios</strong> y <strong>magnetómetros</strong>. Se utilizan para clasificar actividades físicas.</p>
          <p><strong>Actividades detectadas:</strong></p>
          <ul>
            <li><strong>bike</strong>: andar en bicicleta</li>
            <li><strong>climbing</strong>: escalar</li>
            <li><strong>descending</strong>: bajar escaleras o pendientes</li>
            <li><strong>gymbike</strong>: bicicleta estática</li>
            <li><strong>jumping</strong>: saltar</li>
            <li><strong>running</strong>: correr</li>
            <li><strong>standing</strong>: estar de pie</li>
            <li><strong>treadmill</strong>: correr en cinta</li>
            <li><strong>walking</strong>: caminar</li>
          </ul>
        </>
      );
    } else if (infoType === 'audio') {
      content = (
        <p>
          Los datos de audio provienen de <strong>grabaciones de 1 minuto</strong> realizadas a maquinaria en funcionamiento. Estos audios se procesan como <strong>espectrogramas</strong>, y se utilizan para detectar el <strong>estado de la máquina</strong>, clasificando los sonidos en dos categorías: <strong>normales</strong> y <strong>anómalos</strong>.
        </p>
      );
    } else if (infoType === 'video') {
      content = (
        <p>
          Los datos de video representan <strong>secuencias de imágenes etiquetadas</strong> utilizadas para tareas de <strong>detección de objetos</strong>. En este caso, el sistema detecta elementos de seguridad como <strong>casco</strong>, <strong>chaleco</strong> y <strong>mascarilla</strong>.
        </p>
      );
    }
    return content;
  };

  return (
    <div className="container py-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4 className="fw-bold mb-0">Datasets disponibles</h4>
        <div className="d-flex gap-2">
          <button className="btn btn-secondary" onClick={() => setShowModal(true)}>
            <i className="bi bi-plus-circle me-1"></i> Generar nuevo dataset
          </button>

          {/* Botón para subir CSV */}
          <Form.Group controlId="formFileCsvUpload" className="mb-0">
            <Form.Label className="btn btn-outline-secondary mb-0">
              <i className="bi bi-upload me-1"></i> Subir CSV
              <Form.Control
                type="file"
                accept=".csv"
                onChange={handleCsvUpload}
                style={{ display: 'none' }}
              />
            </Form.Label>
          </Form.Group>
        </div>
      </div>

      <div className="mb-3">
        <Button variant="outline-secondary" className="me-2" onClick={() => setInfoType('sensor')}>Sensor</Button>
        <Button variant="outline-secondary" className="me-2" onClick={() => setInfoType('audio')}>Audio</Button>
        <Button variant="outline-secondary" onClick={() => setInfoType('video')}>Video</Button>
      </div>

      {renderInfoModal()}

      <div className="table-responsive">
        {isLoading && (
          <div className="text-center my-4">
            <div className="spinner-border text-secondary" role="status">
              <span className="visually-hidden">Cargando...</span>
            </div>
            <p className="mt-2 mb-0 text-muted">Procesando... por favor espera.</p>
          </div>
        )}

        <table className="table table-bordered align-middle table-hover mt-3">
          <thead className="table-light">
            <tr>
              <th>Nombre</th>
              <th>Tipo</th>
              <th>Tamaño</th>
              <th>Estado</th>
              <th>Fecha de creación</th>
              <th>Acciones</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map(dataset => (
              <tr key={dataset.id}>
                <td>{dataset.name}</td>
                <td>
                  <span className={`badge ${
                    dataset.data_type.toLowerCase() === 'sensor'
                      ? 'bg-info text-dark'
                      : dataset.data_type.toLowerCase() === 'video'
                      ? 'bg-secondary'
                      : 'bg-success'
                  }`}>
                    {dataset.data_type.charAt(0).toUpperCase() + dataset.data_type.slice(1)}
                  </span>
                </td>
                <td>{dataset.sample_count.toLocaleString()} muestras</td>
                <td>{renderStatusBadge(dataset.status)}</td>
                <td>{new Date(dataset.created_at).toLocaleDateString()}</td>
                <td className="text-center">
                  <button
                    className="btn btn-sm btn-outline-primary me-2"
                    onClick={() => {
                      setSelectedDatasetId(dataset.id);
                      setShowVisualizeModal(true);
                    }}
                    title="Visualizar data"
                    disabled={dataset.status !== 'completado'}
                  >
                    <i className="bi bi-bar-chart-line"></i>
                  </button>
                  <button
                    className="btn btn-sm btn-outline-danger"
                    title="Eliminar"
                    onClick={async () => {
                      if (window.confirm(`¿Estás seguro de eliminar el dataset "${dataset.name}"?`)) {
                        try {
                          await axios.delete(`${apiUrl}/datasets/${dataset.id}/`, {
                            headers: {
                              Authorization: `Bearer ${localStorage.getItem('access')}`,
                            },
                          });
                          setDatasets(prev => prev.filter(d => d.id !== dataset.id));
                        } catch (error) {
                          console.error('Error al eliminar dataset:', error);
                          alert('Error al eliminar el dataset');
                        }
                      }
                    }}
                  >
                    <i className="bi bi-trash"></i>
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <VisualizeDataModal
        show={showVisualizeModal}
        onHide={() => setShowVisualizeModal(false)}
        datasetId={selectedDatasetId}
      />

      <GenerateDataModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        onCreated={handleDatasetCreated}
        setLoading={setIsLoading}
      />
    </div>
  );
}

export default Datos;
