import React, { useState, useEffect } from 'react';
import { Card, Alert, Spinner } from 'react-bootstrap';

function YoloVisualization() {
  const [images, setImages] = useState({ train_image: null, val_labels_image: null });
  const [imageError, setImageError] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem("access");

    fetch('http://localhost:8000/api/latest_yolo_image/', {
      headers: token ? {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json"
      } : {},
    })
      .then(async res => {
        if (!res.ok) {
          const errorText = await res.text();
          throw new Error(errorText);
        }
        return res.json();
      })
      .then(data => {
        if (data.train_image || data.val_labels_image) {
          setImages(data);
        } else if (data.error) {
          setImageError(data.error);
        }
      })
      .catch(err => {
        console.error("❌ Error cargando imágenes YOLO:", err);
        setImageError("No se pudieron cargar las imágenes.");
      });
  }, []);

  if (imageError) {
    return <Alert variant="danger">{imageError}</Alert>;
  }

  if (!images.train_image && !images.val_labels_image) {
    return (
      <div className="text-center py-5">
        <Spinner animation="border" variant="primary" />
        <p className="mt-3 text-muted">Cargando imágenes YOLO...</p>
      </div>
    );
  }

  return (
    <div className="d-flex justify-content-center gap-4 flex-wrap mt-4">
      {images.train_image && (
        <Card className="p-2 text-center" style={{ width: "635px" }}>
          <img
            src={`data:image/jpeg;base64,${images.train_image}`}
            alt="Train batch"
            style={{ width: "100%", borderRadius: "8px" }}
          />
          <p className="mt-2 text-muted">Imagen de entrenamiento</p>
        </Card>
      )}
      {images.val_labels_image && (
        <Card className="p-2 text-center" style={{ width: "635px" }}>
          <img
            src={`data:image/jpeg;base64,${images.val_labels_image}`}
            alt="Val labels"
            style={{ width: "100%", borderRadius: "8px" }}
          />
          <p className="mt-2 text-muted">Etiquetas de validación</p>
        </Card>
      )}
    </div>
  );
}

export default YoloVisualization;
