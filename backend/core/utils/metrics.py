import numpy as np
from sklearn.metrics import mean_squared_error

def basic_stats(X_real, X_gen):
    real_flat = X_real.reshape(len(X_real), -1)
    gen_flat = X_gen.reshape(len(X_gen), -1)

    # Asegurar que ambas matrices tengan mismo número de columnas
    min_features = min(real_flat.shape[1], gen_flat.shape[1])
    real_flat = real_flat[:, :min_features]
    gen_flat = gen_flat[:, :min_features]

    real_avg = np.mean(real_flat, axis=0)
    gen_avg = np.mean(gen_flat, axis=0)

    stats = {
        "mean_real": float(np.mean(real_flat)),
        "mean_gen": float(np.mean(gen_flat)),
        "std_real": float(np.std(real_flat)),
        "std_gen": float(np.std(gen_flat)),
        "mse_global": float(mean_squared_error(real_avg, gen_avg))
    }

    return stats
