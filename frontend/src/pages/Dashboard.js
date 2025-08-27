import React, { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { FaDatabase, FaBookOpen, FaChartLine, FaChartBar } from 'react-icons/fa';

function Dashboard() {
  const [stats, setStats] = useState({
    models: 0,
    datasets: 0,
    projects: 0,
    average_accuracy: 0,
    best_accuracy: 0,
  });
  const name = localStorage.getItem('name') || '';
  const [recentProjects, setRecentProjects] = useState([]);

  useEffect(() => {
    const fetchStats = async () => {
      const token = localStorage.getItem('access');
      try {
        const [statsRes, recentRes] = await Promise.all([
          fetch('http://localhost:8000/api/dashboard/stats/', {
            headers: { Authorization: `Bearer ${token}` },
          }),
          fetch('http://localhost:8000/api/dashboard/recent_trainings/', {
            headers: { Authorization: `Bearer ${token}` },
          }),
        ]);
        if (statsRes.ok) setStats(await statsRes.json());
        if (recentRes.ok) setRecentProjects(await recentRes.json());
      } catch (err) {
        console.error('Error cargando dashboard', err);
      }
    };

    fetchStats();
  }, []);

  return (
    <div className="container py-5">
      <div className="mb-5">
        <h1 className="fw-bold mb-2">Panel de Control</h1>
        <p className="text-muted fs-5">Hola, {name}. Bienvenid@. Aquí tienes un resumen de lo más relevante en tu espacio de trabajo.</p>
      </div>

      <div className="row g-4">
  <div className="col-md-6">
    <div className="card h-100 border-0 shadow-sm rounded-4" style={{ backgroundColor: "#eef6fb" }}>
      <div className="card-body">
        <h5 className="card-title text-dark"><FaDatabase className="me-2" /> Generar datos</h5>
        <p className="card-text text-secondary">Accede al generador de datos sintéticos. Configura parámetros y crea nuevos datos con DeepSMOTE.</p>
        <a href="datos" className="btn btn-sm btn-outline-primary">Ir a Generar</a>
      </div>
    </div>
  </div>

  <div className="col-md-6">
    <div className="card h-100 border-0 shadow-sm rounded-4" style={{ backgroundColor: "#f1f8f5" }}>
      <div className="card-body">
        <h5 className="card-title text-dark"><FaBookOpen className="me-2" /> Guía rápida</h5>
        <p className="card-text text-secondary">Aprende a usar la plataforma paso a paso. Ideal para nuevos usuarios.</p>
        <a href="guia" className="btn btn-sm btn-outline-success">Ver Guía</a>
      </div>
    </div>
  </div>

  <div className="col-md-6">
    <div className="card h-100 border-0 shadow-sm rounded-4" style={{ backgroundColor: "#fff7e6" }}>
      <div className="card-body">
        <h5 className="card-title text-dark"><FaChartLine className="me-2" /> Entrenamientos recientes</h5>
        {recentProjects.length > 0 ? (
          <ul className="list-unstyled mb-3">
            {recentProjects.map((proj, idx) => (
              <li key={idx} className="mb-1 text-dark">
                <strong>{proj.nombre}</strong> — {proj.modelo}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-muted">Aún no tienes entrenamientos recientes.</p>
        )}
        <a href="Entrenar" className="btn btn-sm btn-outline-warning text-dark">Ver todos</a>
      </div>
    </div>
  </div>

  <div className="col-md-6">
    <div className="card h-100 border-0 shadow-sm rounded-4" style={{ backgroundColor: "#f4f4fc" }}>
      <div className="card-body">
        <h5 className="card-title text-dark"><FaChartBar className="me-2" /> Estadísticas rápidas</h5>
        <div className="row text-center text-muted">
          <div className="col-6 mb-3">
            <h3 className="fw-bold text-dark">{stats.models}</h3>
            <p className="mb-0">Modelos entrenados</p>
          </div>
          <div className="col-6 mb-3">
            <h3 className="fw-bold text-dark">{stats.projects}</h3>
            <p className="mb-0">Proyectos activos</p>
          </div>
          <div className="col-6 mb-3">
            <h3 className="fw-bold text-dark">{stats.datasets}</h3>
            <p className="mb-0">Datasets generados</p>
          </div>
          <div className="col-6 mb-3">
            <h3 className="fw-bold text-dark">{stats.best_accuracy}%</h3>
            <p className="mb-0">Mejor precisión</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>


      <Outlet />
    </div>
  );
}

export default Dashboard;
