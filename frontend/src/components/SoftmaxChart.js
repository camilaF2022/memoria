import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart,
  LineElement,
  BarElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  ArcElement
} from 'chart.js';

Chart.register(
  LineElement,
  BarElement,
  PointElement,
  LinearScale,
  CategoryScale,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

function SoftmaxChart({ lastPrediction }) {
  if (!lastPrediction || !lastPrediction.probabilities) {
    return <p className="text-muted">Realiza una predicción para visualizar las probabilidades.</p>;
  }

  const labels = Object.keys(lastPrediction.probabilities);
  const dataValues = Object.values(lastPrediction.probabilities).map(p => p * 100);

  const data = {
    labels: labels,
    datasets: [
      {
        label: 'Probabilidad (%)',
        data: dataValues,
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Probabilidad (%)'
        }
      }
    }
  };

  return (
    <div>
      <h6 className="fw-bold">Gráfico de Confianza (Softmax)</h6>
      <Bar data={data} options={options} />
    </div>
  );
}

export default SoftmaxChart;
