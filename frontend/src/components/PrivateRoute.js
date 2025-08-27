import { Navigate, Outlet } from 'react-router-dom';

const PrivateRoute = () => {
  const token = localStorage.getItem('access');

  return token ? <Outlet /> : <Navigate to="/" />;
};

export default PrivateRoute;
