import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function ResetPassword() {
  const [form, setForm] = useState({
    email: '',
    code: '',
    new_password: '',
    confirm_password: ''
  });

  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const storedEmail = localStorage.getItem('reset_email');
    if (storedEmail) {
      setForm(prev => ({ ...prev, email: storedEmail }));
    }
  }, []);

  const handleReset = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    if (form.new_password !== form.confirm_password) {
      setError('❌ Las contraseñas no coinciden');
      return;
    }

    try {
      const res = await fetch('http://localhost:8000/api/password/reset/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      });

      const data = await res.json();

      if (res.ok) {
        setMessage('✅ Contraseña actualizada exitosamente');
        localStorage.removeItem('reset_email');
        setTimeout(() => navigate('/'), 1500);
      } else {
        setError(data.error || '❌ Error al cambiar contraseña');
      }
    } catch (err) {
      setError('❌ Error de red');
    }
  };

  return (
    <div
      className="d-flex justify-content-center align-items-center"
      style={{ height: '100vh', background: 'linear-gradient(to right, #f0f4ff, #ffffff)' }}
    >
      <div className="card shadow-lg border-0" style={{ width: '100%', maxWidth: '500px', borderRadius: '20px' }}>
        <div className="card-body p-5">
          <div className="text-center mb-4">
            <h3 className="fw-bold">Restablecer Contraseña</h3>
            <p className="text-muted">Ingresa el código recibido y tu nueva contraseña</p>
          </div>

          <form onSubmit={handleReset}>
            <div className="mb-3">
              <label className="form-label">Correo electrónico</label>
              <input
                type="email"
                className="form-control form-control-lg"
                placeholder="tucorreo@ejemplo.com"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required
              />
            </div>

            <div className="mb-3">
              <label className="form-label">Código recibido</label>
              <input
                type="text"
                className="form-control form-control-lg"
                placeholder="Código"
                value={form.code}
                onChange={(e) => setForm({ ...form, code: e.target.value })}
                required
              />
            </div>

            <div className="mb-3">
              <label className="form-label">Nueva contraseña</label>
              <input
                type="password"
                className="form-control form-control-lg"
                placeholder="Nueva contraseña"
                value={form.new_password}
                onChange={(e) => setForm({ ...form, new_password: e.target.value })}
                required
              />
            </div>

            <div className="mb-4">
              <label className="form-label">Confirmar nueva contraseña</label>
              <input
                type="password"
                className="form-control form-control-lg"
                placeholder="Confirmar contraseña"
                value={form.confirm_password}
                onChange={(e) => setForm({ ...form, confirm_password: e.target.value })}
                required
              />
            </div>

            {message && <div className="alert alert-success text-center">{message}</div>}
            {error && <div className="alert alert-danger text-center">{error}</div>}

            <button type="submit" className="btn btn-primary btn-lg w-100">Cambiar contraseña</button>
          </form>

          <div className="text-center mt-4">
            <Link to="/" className="text-decoration-none text-primary">Volver al inicio de sesión</Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ResetPassword;
