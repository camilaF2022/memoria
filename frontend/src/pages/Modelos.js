import React, { useState, useEffect } from 'react';
import GenerateModelModal from '../components/GenerateModelModal';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import InfoModal from '../components/InfoModal';

function Modelos() {
  const [showModal, setShowModal] = useState(false);
  const [modelos, setModelos] = useState([]);
  const [hasDatasets, setHasDatasets] = useState(true);
  const [modeloAEditar, setModeloAEditar] = useState(null);
  const apiUrl = "http://localhost:8000/api";
  const navigate = useNavigate();
  const modelosInfo = {
    CNN: {
      title: 'CNN (Convolutional Neural Network)',
      content: 'Las CNN son modelos diseñados para analizar imágenes o datos similares a imágenes, como los espectrogramas. Funcionan detectando patrones visuales como formas o texturas, y por eso son ideales para clasificar datos de sensores transformados en formato visual (2D). En este proyecto, las CNN permiten identificar actividades físicas o estados de una máquina a partir de esas "imágenes de datos".'
    },    
    LSTM: {
      title: 'LSTM (Long Short-Term Memory)',
      content: 'Las LSTM son un tipo especial de red neuronal que se usa para analizar datos que cambian con el tiempo, como señales de sensores o audio. Son buenas para "recordar" lo que pasó antes y detectar patrones en secuencias largas. En este proyecto, permiten clasificar actividades o estados anómalos observando cómo varían las señales en el tiempo.'
    },    
    SVM: {
      title: 'SVM (Support Vector Machine)',
      content: 'Los modelos SVM son algoritmos que aprenden a separar diferentes categorías de datos dibujando una línea (o plano) que mejor los divide. Aunque no son redes neuronales, son muy efectivos cuando se usan con vectores que resumen información, como características extraídas de imágenes o señales.'
    },    
    NaiveBayes: {
      title: 'Naive Bayes',
      content: 'Naive Bayes es un modelo que clasifica cosas según la probabilidad de que pertenezcan a una categoría, usando reglas simples de probabilidad. Es muy rápido y funciona bien cuando los datos se pueden describir con características básicas, como en textos o etiquetas simples.'
    },    
    YOLO: {
      title: 'YOLO (You Only Look Once)',
      content: 'YOLO es un modelo que detecta objetos dentro de una imagen de forma muy rápida. Puede decir qué objetos hay (como un casco o una mascarilla) y dónde están ubicados, todo en una sola pasada. Es ideal para tareas de vigilancia o seguridad en tiempo real.'
    }, 
    KMeans: {
      title: 'KMeans (K-Means Clustering)',
      content: 'KMeans es un modelo que agrupa datos similares sin necesidad de etiquetas. Por ejemplo, puede juntar automáticamente señales de sensores que se parezcan entre sí, creando "grupos" o "categorías" ocultas. Es útil cuando no sabemos de antemano cómo están clasificados los datos.'
    }
    
    
  };
  const [info, setInfo] = useState({ show: false, title: '', content: '' });

  const fetchModelos = async () => {
    try {
      const token = localStorage.getItem('access');

      const dsRes = await axios.get(`${apiUrl}/datasets/`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHasDatasets(dsRes.data.length > 0);

      if (dsRes.data.length > 0) {
        const { data } = await axios.get(`${apiUrl}/models/`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setModelos(data);
      }

    } catch (error) {
      console.error('Error al cargar modelos o datasets:', error);
    }
  };

  useEffect(() => {
    fetchModelos();
  }, []);

  const handleCreated = (nuevoModelo) => {
    if (modeloAEditar) {
      setModelos(prev =>
        prev.map(m => (m.id === nuevoModelo.id ? nuevoModelo : m))
      );
    } else {
      setModelos(prev => [nuevoModelo, ...prev]);
    }
    setModeloAEditar(null); // reset al terminar edición
  };

  const handleDelete = async (modelo) => {
    if (window.confirm(`¿Estás seguro de eliminar el modelo "${modelo.name}"?`)) {
      try {
        await axios.delete(`${apiUrl}/models/${modelo.id}/`, {
          headers: {
            Authorization: `Bearer ${localStorage.getItem('access')}`,
          },
        });
        setModelos(prev => prev.filter(m => m.id !== modelo.id));
      } catch (err) {
        console.error('Error al eliminar modelo:', err);
        alert('No se pudo eliminar el modelo.');
      }
    }
  };

  return (
    <div className="container py-4">
      <h4 className="fw-bold mb-3">Modelos creados</h4>

      {!hasDatasets ? (
        <div className="alert alert-warning text-center">
          <p className="mb-2">🔔 Primero debes generar un dataset antes de crear un modelo.</p>
          <button
            className="btn btn-secondary"
            onClick={() => navigate('/dashboard/datos')}
          >
            Ir a generar datos
          </button>
        </div>
      ) : (
        <>
          <div className="d-flex justify-content-between align-items-center mb-3">
            <p className="mb-0 text-muted">Aquí puedes revisar y gestionar tus modelos generados.</p>
            <button
              className="btn btn-secondary"
              onClick={() => {
                setModeloAEditar(null);
                setShowModal(true);
              }}
            >
              <i className="bi bi-plus-circle me-1"></i> Crear nuevo modelo
            </button>
          </div>
          <div className="d-flex flex-wrap gap-2 mb-3">
  {Object.entries(modelosInfo).map(([key, { title }]) => (
    <button
      key={key}
      className="btn btn-outline-secondary"
      onClick={() => setInfo({ show: true, title, content: modelosInfo[key].content })}
    >
      Info {key}
    </button>
  ))}
</div>

          <div className="table-responsive">
            <table className="table table-bordered align-middle table-hover">
              <thead className="table-light">
                <tr>
                  <th>Nombre del modelo</th>
                  <th>Dataset asociado</th>
                  <th>Tipo de modelo</th>
                  <th>Fecha de creación</th>
                  <th>Última modificación</th>
                  <th className="text-center">Acciones</th>
                </tr>
              </thead>
              <tbody>
                {modelos.length === 0 ? (
                  <tr>
                    <td colSpan="6" className="text-center">No hay modelos creados aún.</td>
                  </tr>
                ) : (
                  modelos.map(modelo => (
                    <tr key={modelo.id}>
                      <td>{modelo.name}</td>
                      <td>{modelo.dataset_name || modelo.dataset}</td>
                      <td>
                        <span className={`badge ${modelo.model_type === 'CNN' ? 'bg-primary' : 'bg-success'}`}>
                          {modelo.model_type}
                        </span>
                      </td>
                      <td>{new Date(modelo.created_at).toLocaleDateString()}</td>
                      <td>{new Date(modelo.updated_at).toLocaleDateString()}</td>
                      <td className="text-center">
                        <button
                          className="btn btn-sm btn-link text-primary me-2"
                          title="Editar"
                          onClick={() => {
                            setModeloAEditar(modelo);
                            setShowModal(true);
                          }}
                        >
                          <i className="bi bi-pencil-square"></i>
                        </button>

                        <button
                          className="btn btn-sm btn-link text-danger"
                          title="Eliminar"
                          onClick={() => handleDelete(modelo)}
                        >
                          <i className="bi bi-trash"></i>
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          <GenerateModelModal
            isOpen={showModal}
            onClose={() => {
              setShowModal(false);
              setModeloAEditar(null);
            }}
            onCreated={handleCreated}
            modeloInicial={modeloAEditar}
          />
          <InfoModal
  show={info.show}
  onHide={() => setInfo({ ...info, show: false })}
  title={info.title}
  content={info.content}
/>

        </>
      )}
    </div>
  );
}

export default Modelos;
