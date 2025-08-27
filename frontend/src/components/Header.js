import React from 'react';
import { useNavigate } from 'react-router-dom'; 
import { FaUserCircle, FaBars } from 'react-icons/fa';

function Header({ toggleSidebar }) {
  const navigate = useNavigate();

  const firstName = localStorage.getItem('name') || '';
  const lastName = localStorage.getItem('lastname') || '';

  const handleLogout = async () => {
    const refreshToken = localStorage.getItem('refresh');
    if (!refreshToken) {
      return;
    }

    try {
      const response = await fetch('http://localhost:8000/api/logout/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refresh: refreshToken }),
      });

      if (response.ok) {
        localStorage.clear();
        navigate('/');
      } else {
        const data = await response.json();
        alert('Error en logout: ' + data.error);
      }
    } catch (error) {
      alert('Error en logout: ' + error.message);
    }
  };

  return (
    <nav className="navbar navbar-dark bg-dark px-3 d-flex justify-content-between">
      <button className="btn btn-dark" onClick={toggleSidebar}>
        <FaBars size={24} className="text-white" />
      </button>
      <span className="navbar-brand mb-0 h1">SynthLearn</span>

      <div className="dropdown">
        <button
          className="btn btn-dark dropdown-toggle d-flex align-items-center"
          type="button"
          id="userMenuButton"
          data-bs-toggle="dropdown"
          aria-expanded="false"
        >
          <FaUserCircle size={24} className="me-2" />
          {firstName} {lastName}
        </button>
        <ul className="dropdown-menu dropdown-menu-end" aria-labelledby="userMenuButton">
          <li><button className="dropdown-item" onClick={() => alert('Ir a configuración')}>Configuración</button></li>
          <li><hr className="dropdown-divider" /></li>
          <li><button className="dropdown-item" onClick={handleLogout}>Cerrar sesión</button></li>
        </ul>
      </div>
    </nav>
  );
}

export default Header;
