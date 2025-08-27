import React, { useState, useEffect } from 'react';
import { Modal, Button, Spinner, Form, Alert } from 'react-bootstrap';
import axios from 'axios';
import Plot from 'react-plotly.js';

function VisualizeDataModal({ show, onHide, datasetId }) {
  const [loading, setLoading] = useState(true);
  const [dataType, setDataType] = useState('');
  const [labels, setLabels] = useState([]);
  const [selectedLabel, setSelectedLabel] = useState('');
  const [dataReal, setDataReal] = useState(null);
  const [dataGenerated, setDataGenerated] = useState(null);
  const [imageReal, setImageReal] = useState('');
  const [imageGenerated, setImageGenerated] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const apiUrl = "http://localhost:8000/api";

  const resetState = () => {
    setLoading(true);
    setLabels([]);
    setSelectedLabel('');
    setDataReal(null);
    setDataGenerated(null);
    setImageReal('');
    setImageGenerated('');
    setErrorMessage('');
  };

  useEffect(() => {
    if (show && datasetId) {
      resetState();
      fetchDatasetInfo();
    } else if (!show) {
      resetState();
    }
  }, [show, datasetId]);

  const fetchDatasetInfo = async () => {
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.get(`${apiUrl}/datasets/${datasetId}/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setDataType(data.data_type);
      if (data.data_type === 'video') {
        fetchComparison(); // directamente
      } else {
        fetchLabels();
      }
    } catch (error) {
      console.error('Error al obtener dataset', error);
      setErrorMessage('Error al obtener información del dataset: ' + error.message);
      setLoading(false);
    }
  };

  const fetchLabels = async () => {
    try {
      const token = localStorage.getItem('access');
      const { data } = await axios.get(`${apiUrl}/datasets/${datasetId}/labels/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setLabels(data);
      if (data.length > 0) {
        setSelectedLabel(data[0]);
      } else {
        setLoading(false);
      }
    } catch (error) {
      console.error('Error al obtener etiquetas', error);
      setErrorMessage('Error al obtener etiquetas: ' + error.message);
      setLoading(false);
    }
  };

  const fetchComparison = async (label = '') => {
    try {
      const token = localStorage.getItem('access');
      let url = `${apiUrl}/datasets/${datasetId}/comparison/`;
      if (label) url += `?label=${label}`;

      const { data } = await axios.get(url, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (dataType === 'video') {
        setImageReal(data.real.image);
        setImageGenerated(data.generated.image);
      } else {
        setDataReal(data.real);
        setDataGenerated(data.generated);
      }
    } catch (error) {
      console.error('Error al obtener comparación', error);
      setErrorMessage('Error al obtener datos: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (dataType !== 'video' && selectedLabel && show) {
      setLoading(true);
      fetchComparison(selectedLabel);
    }
  }, [selectedLabel, datasetId, dataType, show]);

  const plotLayout = (title, isSpectrogram = false) => ({
    title: title,
    autosize: true,
    margin: { l: 40, r: 40, t: 40, b: 40 },
    xaxis: isSpectrogram ? { visible: false } : { title: 'Tiempo' },
    yaxis: isSpectrogram ? { visible: false } : { title: 'Amplitud' },
    plot_bgcolor: '#ffffff',
    paper_bgcolor: '#ffffff',
    showlegend: false
  });

  return (
    <Modal show={show} onHide={onHide} size="xl" centered>
      <Modal.Header closeButton>
        <Modal.Title>Visualización de datos</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {loading ? (
          <div className="text-center p-5">
            <Spinner animation="border" />
            <div className="mt-2">Cargando...</div>
          </div>
        ) : errorMessage ? (
          <Alert variant="danger">{errorMessage}</Alert>
        ) : dataType === 'video' ? (
          <div className="d-flex justify-content-around flex-wrap gap-4">
            <div>
              <h6 className="text-center">REAL</h6>
              <img src={`data:image/jpeg;base64,${imageReal}`} alt="Real" className="img-fluid" style={{ maxHeight: 300 }} />
            </div>
            <div>
              <h6 className="text-center">GENERADO</h6>
              <img src={`data:image/jpeg;base64,${imageGenerated}`} alt="Generado" className="img-fluid" style={{ maxHeight: 300 }} />
            </div>
          </div>
        ) : dataReal && dataGenerated ? (
          <>
            <Form.Group className="mb-3">
              <Form.Label>Etiqueta a comparar:</Form.Label>
              <Form.Select
                value={selectedLabel}
                onChange={e => setSelectedLabel(e.target.value)}
              >
                {labels.map(label => (
                  <option key={label} value={label}>{label}</option>
                ))}
              </Form.Select>
            </Form.Group>

            <div className="mt-4">
              <h5>Espectrograma</h5>
              <div className="d-flex flex-wrap justify-content-between gap-4">
                <div style={{ width: '48%' }}>
                  <h6 className="text-center">REAL</h6>
                  <Plot
                    data={dataReal.spectrogram.data}
                    layout={plotLayout('Espectrograma REAL', true)}
                    style={{ height: '350px' }}
                    config={{ displayModeBar: false }}
                  />
                </div>
                <div style={{ width: '48%' }}>
                  <h6 className="text-center">GENERADO</h6>
                  <Plot
                    data={dataGenerated.spectrogram.data}
                    layout={plotLayout('Espectrograma GENERADO', true)}
                    style={{ height: '350px' }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            </div>

            {/* <div className="mt-5">
              <h5>Temporal</h5>
              <div className="d-flex flex-wrap justify-content-between gap-4">
                <div style={{ width: '48%' }}>
                  <h6 className="text-center">REAL</h6>
                  <Plot
                    data={dataReal.temporal.data}
                    layout={plotLayout('Temporal REAL')}
                    style={{ height: '350px' }}
                    config={{ displayModeBar: false }}
                  />
                </div>
                <div style={{ width: '48%' }}>
                  <h6 className="text-center">GENERADO</h6>
                  <Plot
                    data={dataGenerated.temporal.data}
                    layout={plotLayout('Temporal GENERADO')}
                    style={{ height: '350px' }}
                    config={{ displayModeBar: false }}
                  />
                </div>
              </div>
            </div> */}
            {dataReal.temporal && dataGenerated.temporal && (
  <div className="mt-5">
    <h5>Temporal</h5>
    <div className="d-flex flex-wrap justify-content-between gap-4">
      <div style={{ width: '48%' }}>
        <h6 className="text-center">REAL</h6>
        <Plot
          data={dataReal.temporal.data}
          layout={plotLayout('Temporal REAL')}
          style={{ height: '350px' }}
          config={{ displayModeBar: false }}
        />
      </div>
      <div style={{ width: '48%' }}>
        <h6 className="text-center">GENERADO</h6>
        <Plot
          data={dataGenerated.temporal.data}
          layout={plotLayout('Temporal GENERADO')}
          style={{ height: '350px' }}
          config={{ displayModeBar: false }}
        />
      </div>
    </div>
  </div>
)}

          </>
        ) : (
          <Alert variant="warning">No hay datos disponibles para mostrar.</Alert>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>Cerrar</Button>
      </Modal.Footer>
    </Modal>
  );
}

export default VisualizeDataModal;
