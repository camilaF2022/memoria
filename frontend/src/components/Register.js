import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

function Register() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    confirm_password: '',
  });

  const [errors, setErrors] = useState({});
  const [success, setSuccess] = useState('');

  const validate = () => {
    const newErrors = {};
    if (!formData.first_name.trim()) newErrors.first_name = 'Nombre requerido';
    if (!formData.last_name.trim()) newErrors.last_name = 'Apellido requerido';
    if (!formData.email.includes('@')) newErrors.email = 'Correo inválido';
    if (formData.password.length < 6) newErrors.password = 'La contraseña debe tener al menos 6 caracteres';
    if (formData.password !== formData.confirm_password) newErrors.confirm_password = 'Las contraseñas no coinciden';
    return newErrors;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const errs = validate();
    if (Object.keys(errs).length > 0) {
      setErrors(errs);
      return;
    }

    try {
      const res = await fetch('http://localhost:8000/api/register/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (res.ok) {
        setSuccess('✅ Registro exitoso. Ahora puedes iniciar sesión.');
        setFormData({
          first_name: '',
          last_name: '',
          email: '',
          password: '',
          confirm_password: ''
        });
        setErrors({});
        setTimeout(() => navigate('/'), 2000);
      } else {
        const data = await res.json();
        setErrors({ server: data.error || 'Error al registrar' });
      }
    } catch (err) {
      setErrors({ server: 'Error de red. Intenta más tarde.' });
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
            <h3 className="fw-bold">Registro de Usuario</h3>
            <p className="text-muted">Crea tu cuenta para comenzar</p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <input
                type="text"
                className="form-control form-control-lg"
                placeholder="Nombre"
                value={formData.first_name}
                onChange={(e) => setFormData({ ...formData, first_name: e.target.value })}
              />
              {errors.first_name && <small className="text-danger">{errors.first_name}</small>}
            </div>

            <div className="mb-3">
              <input
                type="text"
                className="form-control form-control-lg"
                placeholder="Apellido"
                value={formData.last_name}
                onChange={(e) => setFormData({ ...formData, last_name: e.target.value })}
              />
              {errors.last_name && <small className="text-danger">{errors.last_name}</small>}
            </div>

            <div className="mb-3">
              <input
                type="email"
                className="form-control form-control-lg"
                placeholder="Correo electrónico"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
              />
              {errors.email && <small className="text-danger">{errors.email}</small>}
            </div>

            <div className="mb-3">
              <input
                type="password"
                className="form-control form-control-lg"
                placeholder="Contraseña"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
              />
              {errors.password && <small className="text-danger">{errors.password}</small>}
            </div>

            <div className="mb-4">
              <input
                type="password"
                className="form-control form-control-lg"
                placeholder="Confirmar contraseña"
                value={formData.confirm_password}
                onChange={(e) => setFormData({ ...formData, confirm_password: e.target.value })}
              />
              {errors.confirm_password && <small className="text-danger">{errors.confirm_password}</small>}
            </div>

            {errors.server && <div className="alert alert-danger text-center">{errors.server}</div>}
            {success && <div className="alert alert-success text-center">{success}</div>}

            <button type="submit" className="btn btn-primary btn-lg w-100">Registrarse</button>
          </form>

          <div className="text-center mt-4">
            <span>¿Ya tienes cuenta? <Link to="/" className="text-primary">Inicia sesión</Link></span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Register;
