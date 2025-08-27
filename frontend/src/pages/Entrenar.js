import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Button } from 'react-bootstrap';
import TrainModelModal from '../components/TrainModelModal';
import { useNavigate } from 'react-router-dom';
import TrainingParamsInfoModal from '../components/TrainingParamsInfoModal';

function Entrenar() {
  const [showTrainModal, setShowTrainModal] = useState(false);
  const [trainings, setTrainings] = useState([]);
  const [isTrainingInProgress, setIsTrainingInProgress] = useState(false);
  const [modelos, setModelos] = useState([]);
  const [showParamsInfo, setShowParamsInfo] = useState(false);

  const apiUrl = "http://localhost:8000/api";
  const navigate = useNavigate();

  const fetchTrainings = async () => {
    try {
      const token = localStorage.getItem('access');

      // Obtener modelos
      const modelosRes = await axios.get(`${apiUrl}/models/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setModelos(modelosRes.data);

      const trainingsRes = await axios.get(`${apiUrl}/trains/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setTrainings(trainingsRes.data);

      const enProceso = trainingsRes.data.some(train =>
        ['pendiente', 'en progreso'].includes(train.status)
      );
      setIsTrainingInProgress(enProceso);

    } catch (error) {
      console.error('Error al cargar entrenamientos o modelos:', error);
    }
  };

  const handleRowClick = (train) => {
    localStorage.setItem('showEntrenamientosTab', 'true');
    navigate(`/dashboard/mis-entrenamientos/${train.id}`, { state: { train } });
  };
  
  

  useEffect(() => {
    fetchTrainings();
    const interval = setInterval(() => {
      fetchTrainings();
    }, 5000); 
    return () => clearInterval(interval);
  }, []);

  const handleTrained = (nuevoEntrenamiento) => {
    setTrainings(prev => [nuevoEntrenamiento, ...prev]);
  };

  return (
    <div className="container py-4">
      <div className="d-flex justify-content-between align-items-center mb-3">
        <h4 className="fw-bold mb-0">Modelos Entrenados</h4>
        {modelos.length > 0 && (
          <Button
            className="btn btn-secondary"
            variant="primary"
            onClick={() => setShowTrainModal(true)}
          >
            <i className="bi bi-play-circle me-1"></i> Entrenar nuevo modelo
          </Button>
        )}
      </div>

      <div className="table-responsive">
  {modelos.length === 0 ? (
    <div className="text-center py-5">
      <h5 className="text-muted mb-3">Aún no has creado ningún modelo</h5>
      <p className="text-muted">Primero necesitas generar un modelo para poder entrenarlo.</p>
      <Button
        variant="outline-secondary"
        onClick={() => navigate('/dashboard/modelos')}
      >
        <i className="bi bi-plus-circle me-1"></i> Ir a generar modelo
      </Button>
    </div>
  ) : (

    
    <><div className="mb-3 d-flex gap-2">
    <Button variant="outline-secondary" onClick={() => setShowParamsInfo(true)}>
      <i className="bi bi-info-circle me-1"></i> Ver parámetros
    </Button>
  </div>
  
      <table className="table table-bordered align-middle table-hover">
        <thead className="table-light">
          <tr>
            <th>Nombre del entrenamiento</th>
            <th>Nombre del modelo</th>
            <th>Tipo</th>
            <th>Épocas</th>
            <th>Tamaño del Batch</th>
            <th>Precisión</th>
            <th>Fecha</th>
            <th>Estado</th>
            <th className="text-center">Acciones</th>
          </tr>
        </thead>
        <tbody>
          {trainings.length === 0 ? (
            <tr>
              <td colSpan="9" className="text-center py-4 text-muted">
                Aún no has entrenado ningún modelo. Haz clic en "Entrenar nuevo modelo" para comenzar.
              </td>
            </tr>
          ) : (
            trainings.map(train => (
              <tr
                key={train.id}
                style={{ cursor: 'pointer' }}
                onClick={() => handleRowClick(train)}
              >
                <td>{train.training_name}</td>
                <td>{train.model_name}</td>
                <td>
                  <span className={`badge ${train.model_type === 'CNN' ? 'bg-primary' : 'bg-success'}`}>
                    {train.model_type}
                  </span>
                </td>
                <td>{train.epochs}</td>
                <td>{train.batch_size}</td>
                <td>{(train.accuracy * 100).toFixed(2)}%</td>
                <td>{new Date(train.created_at).toLocaleDateString()}</td>
                <td>
                  <span className={`badge ${
                    train.status === 'completado' ? 'bg-success' :
                    train.status === 'en progreso' ? 'bg-warning text-dark' :
                    train.status === 'pendiente' ? 'bg-secondary' :
                    'bg-danger'
                  }`}>
                    {train.status}
                  </span>
                </td>
                <td className="text-center">
                  {train.status === 'completado' && (
                    <Button
                      variant="outline-success"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRowClick(train);
                      }}
                    >
                      <i className="bi bi-bar-chart"></i> Analizar
                    </Button>
                  )}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </>
  )}
</div>



      <TrainModelModal
        isOpen={showTrainModal}
        onClose={() => setShowTrainModal(false)}
        onTrained={handleTrained}
        onStartTraining={() => setIsTrainingInProgress(true)}
      />
      <TrainingParamsInfoModal
  show={showParamsInfo}
  onHide={() => setShowParamsInfo(false)}
/>


      {isTrainingInProgress && (
        <div className="text-center py-3">
          <div className="spinner-border text-primary" role="status" />
          <p className="mt-2 mb-0 text-muted">Entrenando modelo... esto puede tardar unos minutos.</p>
        </div>
      )}
    </div>
  );
}

export default Entrenar;
