import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';

function ForgotPassword() {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSendCode = async (e) => {
    e.preventDefault();
    setMessage('');
    setError('');

    try {
      const res = await fetch('http://localhost:8000/api/password/send_code/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email }),
      });

      const data = await res.json();
      console.log(data)
      if (res.ok) {
        setMessage('✅ Código enviado al correo');
        localStorage.setItem('reset_email', email);
        setTimeout(() => navigate('/restablecer-contraseña'), 1000);
      } else {
        setError(data.error || '❌ Error al enviar el código');
      }
    } catch (err) {
      setError('❌ Error de red. Intenta más tarde.');
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
            <h3 className="fw-bold">¿Olvidaste tu contraseña?</h3>
            <p className="text-muted">Te enviaremos un código a tu correo electrónico</p>
          </div>

          <form onSubmit={handleSendCode}>
            <div className="mb-4">
              <label className="form-label">Correo electrónico</label>
              <input
                type="email"
                className="form-control form-control-lg"
                placeholder="ejemplo@correo.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            {message && <div className="alert alert-success text-center">{message}</div>}
            {error && <div className="alert alert-danger text-center">{error}</div>}

            <button type="submit" className="btn btn-primary btn-lg w-100">Enviar código</button>
          </form>

          <div className="text-center mt-4">
            <Link to="/" className="text-decoration-none text-primary">
              Volver al inicio de sesión
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ForgotPassword;
