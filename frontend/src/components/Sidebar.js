import React from 'react';
import { NavLink, useLocation, useNavigate } from 'react-router-dom';
import { FaUserCircle, FaHome, FaCog, FaCogs, FaHockeyPuck, FaDatabase, FaBookOpen, FaChartLine } from 'react-icons/fa';

function Sidebar({ isCompact }) {
  const location = useLocation();
  const navigate = useNavigate();
  const showEntrenamientos = localStorage.getItem('showEntrenamientosTab') === 'true';
  const firstName = localStorage.getItem('name') || '';
  const lastName = localStorage.getItem('lastname') || '';

  const modulos = [
    { nombre: 'Resumen', ruta: 'dashboard', icon: <FaHome size={20} /> },
    { nombre: 'Mis Datos', ruta: 'datos', icon: <FaDatabase size={20} /> },
    { nombre: 'Mis Modelos', ruta: 'modelos', icon: <FaCogs size={20} /> },
    { nombre: 'Entrenar', ruta: 'entrenar', icon: <FaHockeyPuck size={20} /> },
    ...(showEntrenamientos
      ? [{
          nombre: 'Mis Entrenamientos',
          ruta: 'entrenamiento',
          icon: <FaChartLine size={20} />,
          disabled: true  // <- aquí
        }]
      : []),
    
    { nombre: 'Guía Rápida', ruta: 'guia', icon: <FaBookOpen size={20} /> },
  ];
 // { nombre: 'Configuración', ruta: 'configuracion', icon: <FaCog size={20} /> },


  const handleClick = (modulo, e) => {
    e.preventDefault();
    if (modulo.disabled) return;
    if (location.pathname !== `/dashboard/${modulo.ruta}`) {
      navigate(`/dashboard/${modulo.ruta}`);
    }
  };
  

  return (
    <div
      className="bg-light border-end vh-100 p-3 d-flex flex-column"
      style={{
        width: isCompact ? '70px' : '270px',
        transition: 'width 0.3s ease',
        height: '100vh',
      }}
    >
      <div
        className={`d-flex flex-column align-items-center mb-5 mt-4`}
        style={{
          visibility: isCompact ? 'hidden' : 'visible',
          opacity: isCompact ? 0 : 1,
          transition: 'opacity 0.3s ease, visibility 0.3s ease'
        }}
      >
        <FaUserCircle size={80} className="text-secondary mb-3" />
        <div className="fw-bold text-dark">{firstName} {lastName}</div>
      </div>

      <ul className="nav flex-column w-100 flex-grow-1">
        {modulos.map((modulo, i) => (
          <React.Fragment key={i}>
            <li className="nav-item">
              <a
                href="#"
                className={`nav-link px-3 py-2 rounded text-dark ${location.pathname === `/dashboard/${modulo.ruta}` ? 'bg-secondary text-white' : 'hover-link'}`}
                onClick={(e) => handleClick(modulo, e)}

              >
                <div className="d-flex align-items-center">
                  {modulo.icon}
                  {!isCompact && <span className="ms-2">{modulo.nombre}</span>}
                </div>
              </a>
            </li>
            {i < modulos.length - 1 && <hr className="my-2" />}
          </React.Fragment>
        ))}
      </ul>

      <style>{`
        .hover-link:hover {
          background-color: #e0e0e0;
          text-decoration: none;
        }
      `}</style>
    </div>
  );
}

export default Sidebar;
