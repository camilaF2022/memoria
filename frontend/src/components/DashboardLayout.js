import React from 'react';
import Header from './Header';
import Sidebar from './Sidebar';
import { Outlet } from 'react-router-dom';
function DashboardLayout({ children, toggleSidebar, isSidebarCompact }) {
  return (
    <div>
      <Header toggleSidebar={toggleSidebar} />
      <div className="d-flex">
        <Sidebar isCompact={isSidebarCompact} />
        <div className="p-4 w-100">
          {children}
          <Outlet /> 
        </div>
      </div>
    </div>
  );
}

export default DashboardLayout;
