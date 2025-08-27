import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

function ModelSummary({ trainId }) {
  const [metrics, setMetrics] = useState(null);
  const apiUrl = "http://localhost:8000/api";

  useEffect(() => {
    const fetchMetrics = async () => {
      if (!trainId) return;
      try {
        const token = localStorage.getItem('access');
        const { data } = await axios.get(`${apiUrl}/get_model_metrics/?train_id=${trainId}`, {
          headers: { Authorization: `Bearer ${token}` }
        });

        const { accuracy, loss, confusion_matrix, class_names } = data;

        if (Array.isArray(confusion_matrix) && confusion_matrix.length > 0) {
          setMetrics({
            accuracy,
            loss,
            confusion_matrix,
            class_names: Array.isArray(class_names) ? class_names : null
          });
        } else {
          setMetrics(null);
        }

      } catch (error) {
        console.error('Error al obtener métricas:', error);
        setMetrics(null);
      }
    };
    fetchMetrics();
  }, [trainId]);

  const renderHeatmap = (matrix, labels) => {
    if (!matrix || !labels) return <p>No hay datos para el heatmap.</p>;

    return (
      <Plot
        data={[
          {
            z: matrix,
            x: labels,
            y: labels,
            type: 'heatmap',
            colorscale: 'Magma',
            showscale: true,
            hoverongaps: false
          }
        ]}
        layout={{
          title: '🔍 Matriz de Confusión',
          xaxis: {
            title: 'Predicción',
            automargin: true
          },
          yaxis: {
            title: 'Real',
            automargin: true
          },
          height: 500
        }}
      />
    );
  };

  return (
    <div>
      <h6 className="fw-bold mb-2">Resumen del Modelo</h6>
      {metrics ? (
        <>
          <p><strong>Precisión Final:</strong> {(metrics.accuracy * 100).toFixed(2)}%</p>
          <p><strong>Pérdida Final:</strong> {metrics.loss.toFixed(4)}</p>
          <h6 className="mt-3">Matriz de Confusión</h6>
          {renderHeatmap(metrics.confusion_matrix, metrics.class_names)}
        </>
      ) : (
        <p className="text-muted">Cargando métricas del modelo...</p>
      )}
    </div>
  );
}

export default ModelSummary;

