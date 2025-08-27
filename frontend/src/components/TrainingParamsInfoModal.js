import React from 'react';
import { Modal, Button } from 'react-bootstrap';

function TrainingParamsInfoModal({ show, onHide }) {
  return (
    <Modal show={show} onHide={onHide} centered size="xl">
      <Modal.Header closeButton>
        <Modal.Title>Parámetros de Entrenamiento</Modal.Title>
      </Modal.Header>
      <Modal.Body>
 

  <h6 className="mt-3"> CNN y LSTM (modelos de sensores o audio)</h6>
  <ul>
    <li><strong>Épocas (epochs):</strong> Cantidad de veces que el modelo revisará todos los datos. Más épocas pueden mejorar el aprendizaje, pero también tardan más.</li>
    <li><strong>Batch Size:</strong> Número de ejemplos que el modelo analiza al mismo tiempo antes de ajustar su "memoria".</li>
  </ul>

  <h6 className="mt-3">YOLO (detección de objetos en imágenes)</h6>
  <ul>
    <li><strong>Épocas (epochs):</strong> Igual que en otros modelos: más épocas, más entrenamiento.</li>
    <li><strong>imgsz:</strong> Tamaño de las imágenes que el modelo verá (por ejemplo, 640 significa 640x640 píxeles).</li>
    <li><strong>lr0:</strong> Velocidad con la que el modelo ajusta lo que aprende. Si es muy alta, puede aprender mal; si es muy baja, puede ser lento.</li>
  </ul>

  <h6 className="mt-3">Naive Bayes</h6>
  <ul>
    <li><strong>test_split:</strong> Parte de los datos que se reserva para verificar qué tan bien aprendió el modelo (por ejemplo, 0.2 significa 20%).</li>
    <li><strong>var_smoothing:</strong> Ayuda a evitar errores matemáticos cuando los datos son muy pequeños o extremos.</li>
  </ul>

  <h6 className="mt-3">SVM</h6>
  <ul>
    <li><strong>kernel:</strong> Forma en que el modelo transforma los datos para separarlos mejor. Puede ser <code>linear</code> (una línea recta) o <code>rbf</code> (más flexible).</li>
    <li><strong>C:</strong> Qué tan estricto es el modelo al clasificar. Un valor alto intenta clasificar todo perfectamente, pero puede ser sensible al ruido.</li>
    <li><strong>test_split:</strong> Porcentaje de datos usados para validar los resultados.</li>
  </ul>

  <h6 className="mt-3">KMeans</h6>
  <ul>
    <li><strong>n_clusters:</strong> Cuántos grupos (o categorías) queremos que el modelo encuentre automáticamente en los datos.</li>
  </ul>
</Modal.Body>

      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Cerrar
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default TrainingParamsInfoModal;
