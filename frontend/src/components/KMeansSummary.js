import React from 'react';

function KMeansSummary({ plotPath }) {
  let fullUrl = null;

  if (plotPath) {
    // Encuentra desde "/media" hacia adelante
    const mediaIndex = plotPath.indexOf('/media');
    if (mediaIndex !== -1) {
      const relativePath = plotPath.slice(mediaIndex);
      fullUrl = `http://localhost:8000${relativePath}`;
    }
  }

  return (
    <div className="text-center">
      <h5 className="fw-bold mb-2">📊 Visualización K-Means</h5>
      {fullUrl ? (
        <img
          src={fullUrl}
          alt="Gráfico KMeans"
          className="img-fluid rounded shadow"
          style={{ maxWidth: '100%', maxHeight: '500px' }}
        />
      ) : (
        <p>No hay gráfico disponible.</p>
      )}
    </div>
  );
}

export default KMeansSummary;
