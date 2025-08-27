import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { getProtectedData } from '../api';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [protectedData, setProtectedData] = useState(null);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();

    if (!username || !password) {
      alert("Por favor, completa todos los campos.");
      return;
    }

    const response = await fetch('http://localhost:8000/api/token/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username, password }),
    });

    const data = await response.json();

    if (response.ok) {
      localStorage.setItem('access', data.access);
      localStorage.setItem('refresh', data.refresh);
      localStorage.setItem('name', data.first_name);
      localStorage.setItem('lastname', data.last_name);

      const result = await getProtectedData();
      setProtectedData(result);

      navigate('/dashboard/dashboard');
    } else {
      alert("Login fallido: Verifica tus credenciales");
    }
  };

  return (
    <div
      className="d-flex justify-content-center align-items-center"
      style={{
        height: '100vh',
        background: 'linear-gradient(to right, #e3f2fd, #ffffff)',
      }}
    >
      <div className="card shadow-lg border-0" style={{ width: '100%', maxWidth: '500px', borderRadius: '20px' }}>
        <div className="card-body p-5">
          <div className="text-center mb-4">
            <div className="mb-3">
              <i className="bi bi-person-circle" style={{ fontSize: '3rem', color: '#0d6efd' }}></i>
            </div>
            <h2 className="fw-bold">Bienvenido a SynthLearn</h2>
            <p className="text-muted">Inicia sesión para continuar</p>
          </div>

          <form onSubmit={handleLogin}>
            <div className="mb-4">
              <label htmlFor="username" className="form-label">Correo electrónico</label>
              <input
                type="email"
                className="form-control form-control-lg"
                id="username"
                placeholder="tucorreo@ejemplo.com"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>

            <div className="mb-4">
              <label htmlFor="password" className="form-label">Contraseña</label>
              <input
                type="password"
                className="form-control form-control-lg"
                id="password"
                placeholder="********"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>

            <button type="submit" className="btn btn-primary btn-lg w-100">
              Iniciar sesión
            </button>
          </form>

          <div className="text-center mt-4">
            <Link to="/olvide-contraseña" className="d-block mb-2 text-decoration-none text-primary">
              ¿Olvidaste tu contraseña?
            </Link>
            <span className="text-muted">¿No tienes cuenta? <Link to="/registro">Regístrate aquí</Link></span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Login;
