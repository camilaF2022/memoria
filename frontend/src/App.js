import React, {useState} from 'react';
import { BrowserRouter as Router, Route, Routes, data } from 'react-router-dom';  
import 'bootstrap/dist/css/bootstrap.min.css';  
import 'bootstrap/dist/js/bootstrap.bundle.min.js';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import DashboardLayout from './components/DashboardLayout';
import Datos from './pages/Datos';
import Modelos from './pages/Modelos'
import Entrenar from './pages/Entrenar'
import Entrenamiento from './pages/Entrenamiento'
import Guia from './pages/Guia'
import Configuracion from './pages/Configuracion'
import PrivateRoute from './components/PrivateRoute';
import ModeloInteractivo from './pages/Entrenamiento';
import { useParams } from 'react-router-dom';
import Register from './components/Register';
import ForgotPassword from './components/ForgotPassword';
import ResetPassword from './components/ResetPassword';

function App() {
  const [isSidebarCompact, setIsSidebarCompact] = useState(false);

  const toggleSidebar = () => {
    setIsSidebarCompact(!isSidebarCompact);
  };

  

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/registro" element={<Register />} />
        <Route path="/olvide-contraseña" element={<ForgotPassword />} />
        <Route path="/restablecer-contraseña" element={<ResetPassword />} />
        <Route element={<PrivateRoute />}>
          <Route 
            path="/dashboard" 
            element={<DashboardLayout toggleSidebar={toggleSidebar} isSidebarCompact={isSidebarCompact} />}
          >
            <Route path='dashboard' element={<Dashboard />} />
            <Route path="datos" element={<Datos />} /> 
            <Route path="modelos" element={<Modelos />} />
            <Route path="entrenar" element={<Entrenar />} />
            <Route path="entrenamiento" element={<Entrenamiento />} />
            <Route path="mis-entrenamientos/:trainId" element={<ModeloInteractivo />} />
            <Route path="guia" element={<Guia />} />
            <Route path="configuracion" element={<Configuracion />} />
          </Route>
        </Route>
      </Routes>
    </Router>
  );
}


export default App;
