import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Spinner, Alert } from 'react-bootstrap';
import { useParams, useLocation } from 'react-router-dom';
import YoloVisualization from '../components/YoloVisualization';
import SoftmaxChart from '../components/SoftmaxChart';
import NoiseTest from '../components/NoiseTest';
import LabelTester from '../components/LabelTester';
import ModelSummary from '../components/ModelSummary';
import KMeansSummary from '../components/KMeansSummary';

function ModeloInteractivo() {
  const { trainId } = useParams();
  const location = useLocation();
  const train = location.state?.train;

  const [lastPrediction, setLastPrediction] = useState(null);
  const [modelType, setModelType] = useState(train?.model_type?.toLowerCase() || null);
  const [loading, setLoading] = useState(!train);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    if (!train && trainId) {
      setLoading(true);
      const token = localStorage.getItem('access');
      fetch(`/api/trains/${trainId}/`, {
        headers: token
          ? {
              Authorization: `Bearer ${token}`,
              'Content-Type': 'application/json',
            }
          : {},
      })
        .then(async res => {
          const text = await res.text();

          try {
            const data = JSON.parse(text);
            const tipo = data.model_type?.toLowerCase();
            console.log("🧪 Tipo de modelo recibido del backend:", tipo);
            setModelType(tipo);
          } catch (err) {
            console.error("❌ Respuesta no es JSON válido:", text);
            setLoadError("El servidor devolvió una respuesta inválida.");
          } finally {
            setLoading(false);
          }
        })
        .catch(err => {
          console.error("❌ Error al obtener el modelo:", err);
          setLoadError("No se pudo cargar el modelo.");
          setLoading(false);
        });
    }
  }, [trainId, train]);

  if (loading) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" variant="primary" />
        <p className="mt-3 text-muted">Cargando modelo...</p>
      </div>
    );
  }

  if (loadError) {
    return (
      <div className="container py-5">
        <Alert variant="danger">{loadError}</Alert>
      </div>
    );
  }

  if (modelType === "kmeans") {
    return (
      <div className="container py-3">
        <h4 className="fw-bold mb-3 text-center">📊 Análisis de Clustering KMeans</h4>
        <Row>
          <Col>
            <Card className="p-3">
            <KMeansSummary plotPath={train?.combined_data?.plot_path} />
            </Card>
          </Col>
        </Row>
      </div>
    );
  }
  
  if (modelType === "yolo") {
    return (
      <div className="container py-3">
        <h4 className="fw-bold mb-3 text-center">📦 Visualización de detección YOLO</h4>
        <YoloVisualization trainId={trainId} />
      </div>
    );
  }
  

  return (
    <div className="container py-3">
      <h4 className="fw-bold mb-3 text-center">🧪 Análisis Interactivo del Modelo Entrenado</h4>

      <Row>
        <Col>
          <Card className="p-3">
            <ModelSummary trainId={trainId} />
          </Card>
        </Col>
      </Row>

      <Row className="mb-3">
        <Col md={6} className="mb-3">
          <Card className="h-100 p-3">
            <LabelTester trainId={trainId} setLastPrediction={setLastPrediction} />
          </Card>
        </Col>

        <Col md={6} className="mb-3">
          <Card className="h-100 p-3">
            <NoiseTest trainId={trainId} setLastPrediction={setLastPrediction} />
          </Card>
        </Col>
      </Row>

      <Row className="mb-3">
        <Col>
          <Card className="p-3">
            <SoftmaxChart trainId={trainId} lastPrediction={lastPrediction} />
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default ModeloInteractivo;
