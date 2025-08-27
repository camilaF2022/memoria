import React from 'react';

function Guia() {
  return (
    <div className="container py-4">
      <h3 className="fw-bold mb-4">Guía Rápida de Uso</h3>

      <div className="mb-4">
        <h5>1. Datos</h5>
        <p>
          Este módulo te permite cargar tus propios archivos o generar datasets sintéticos que luego podrás usar en la creación y entrenamiento de modelos.
        </p>
        <ul>
          <li><strong>Subir CSV:</strong> puedes importar archivos con datos de sensores, audio o video. Asegúrate de que estén correctamente formateados según el tipo correspondiente.</li>
          <li><strong>Generar dataset:</strong> puedes crear nuevos datasets sintéticos aplicando técnicas como DeepSMOTE (en sensores o audio) o aumentación (en imágenes tipo YOLO).</li>
          <li><strong>Información por tipo:</strong> utiliza los botones “Sensor”, “Audio” o “Video” para acceder a guías específicas que explican el formato esperado y cómo se procesan internamente.</li>
        </ul>
        <p className="text-muted">Es necesario tener al menos un dataset para crear modelos.</p>
      </div>

      <div className="mb-4">
        <h5>2. Modelos</h5>
        <p>
          En esta sección defines los modelos que deseas entrenar, especificando su tipo y el dataset sobre el cual se entrenarán.
        </p>
        <ul>
          <li><strong>Crear modelo:</strong> selecciona un dataset, asigna un nombre y elige el tipo de modelo que deseas usar (CNN, LSTM, SVM, Naive Bayes, YOLO o KMeans).</li>
          <li><strong>Editar modelo:</strong> si necesitas modificar un modelo antes de entrenarlo, puedes hacerlo desde el listado con el ícono ✎.</li>
          <li><strong>Aprender sobre los modelos:</strong> cada modelo tiene un botón “Info” con una explicación sobre su funcionamiento y casos recomendados de uso.</li>
        </ul>
        <p className="text-muted">El tipo de modelo debe ser coherente con el tipo de dato (por ejemplo, YOLO para imágenes, LSTM para secuencias).</p>
      </div>

      <div className="mb-4">
        <h5>3.  Entrenar</h5>
        <p>
          Aquí puedes ejecutar el proceso de entrenamiento, monitorear el estado y consultar los resultados una vez finalizado.
        </p>
        <ul>
          <li><strong>Entrenar nuevo modelo:</strong> selecciona el modelo y define parámetros como número de épocas, tamaño de batch y otros según el tipo de algoritmo.</li>
          <li><strong>Ver historial:</strong> se listan todos los entrenamientos realizados, indicando su estado (pendiente, en progreso, completado o fallido) y métricas clave como la precisión.</li>
          <li><strong>Parámetros de entrenamiento:</strong> puedes acceder a la configuración detallada utilizada en cada ejecución mediante el botón “Ver parámetros”.</li>
        </ul>
        <p className="text-muted">Los entrenamientos se ejecutan en segundo plano. Puedes navegar por la plataforma mientras esperas.</p>
      </div>

      <div className="mb-4">
        <h5>4.  Análisis Interactivo</h5>
        <p>
          Este módulo te permite evaluar el rendimiento de un modelo entrenado mediante visualizaciones y pruebas personalizadas.
        </p>
        <ul>
          <li><strong>YOLO:</strong> muestra detecciones de objetos sobre las imágenes cargadas, indicando clases y bounding boxes.</li>
          <li><strong>Modelos de clasificación (CNN / LSTM / SVM / Naive Bayes):</strong> puedes probar el modelo con entradas nuevas, ver la distribución de probabilidades (softmax), introducir ruido, y ver su respuesta ante datos específicos.</li>
          <li><strong>KMeans:</strong> permite explorar cómo se agrupan los datos en el espacio latente sin necesidad de etiquetas, útil para análisis exploratorio.</li>
        </ul>
        <p>Accede desde la sección <strong>“Modelos entrenados”</strong> haciendo clic en el botón “Analizar”.</p>
      </div>
    </div>
  );
}

export default Guia;
