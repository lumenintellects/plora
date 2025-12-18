# Value-add experiment – summary

## Domain: arithmetic

| Cell | r | scheme | ΔNLL | p | 95% CI | split | latency(ms) | guard |
|---|---|---|---|---|---|---|---|---|
| trained_seed41 | 4 | all | **-0.720** | 2.658e-43 | [-0.764, -0.674] | validation | 221.5 |  |
| trained_seed42 | 4 | all | **-0.720** | 2.658e-43 | [-0.764, -0.674] | validation | 215.8 |  |
| trained_seed43 | 4 | all | **-0.720** | 2.658e-43 | [-0.764, -0.674] | validation | 217.7 |  |
| trained_seed41 | 4 | attention | **-0.989** | 2.049e-43 | [-1.026, -0.954] | validation | 173.4 |  |
| trained_seed42 | 4 | attention | **-0.967** | 2.049e-43 | [-1.004, -0.932] | validation | 171.3 |  |
| trained_seed43 | 4 | attention | **-0.988** | 2.049e-43 | [-1.027, -0.954] | validation | 177.6 |  |
| trained_seed41 | 4 | mlp | **-1.215** | 2.049e-43 | [-1.257, -1.177] | validation | 173.5 |  |
| trained_seed42 | 4 | mlp | **-1.223** | 2.049e-43 | [-1.265, -1.184] | validation | 175.2 |  |
| trained_seed43 | 4 | mlp | **-1.172** | 2.049e-43 | [-1.212, -1.134] | validation | 179.9 |  |
| trained_seed41 | 8 | all | **-0.837** | 2.200e-43 | [-0.885, -0.790] | validation | 261.5 | ⚠ |
| trained_seed42 | 8 | all | **-0.935** | 2.049e-43 | [-0.975, -0.894] | validation | 269.5 | ⚠ |
| trained_seed43 | 8 | all | **-0.575** | 2.761e-42 | [-0.623, -0.528] | validation | 281.2 | ⚠ |
| trained_seed41 | 8 | attention | **-0.868** | 2.049e-43 | [-0.908, -0.830] | validation | 176.5 |  |
| trained_seed42 | 8 | attention | **-0.529** | 7.883e-42 | [-0.571, -0.485] | validation | 170.0 |  |
| trained_seed43 | 8 | attention | **-1.043** | 2.049e-43 | [-1.082, -1.007] | validation | 172.1 |  |
| trained_seed41 | 8 | mlp | **-1.203** | 2.049e-43 | [-1.245, -1.164] | validation | 172.7 |  |
| trained_seed42 | 8 | mlp | **-0.846** | 2.049e-43 | [-0.886, -0.807] | validation | 181.3 |  |
| trained_seed43 | 8 | mlp | **-1.169** | 2.049e-43 | [-1.211, -1.129] | validation | 169.7 |  |
| trained_seed41 | 16 | all | **-0.832** | 2.098e-43 | [-0.872, -0.789] | validation | 260.1 | ⚠ |
| trained_seed42 | 16 | all | +0.088 | 4.696e-05 | [0.035, 0.134] | validation | 256.6 | ⚠ |
| trained_seed43 | 16 | all | **-0.843** | 2.049e-43 | [-0.888, -0.797] | validation | 265.8 | ⚠ |
| trained_seed41 | 16 | attention | **-0.746** | 2.174e-43 | [-0.790, -0.702] | validation | 183.3 |  |
| trained_seed42 | 16 | attention | **-0.844** | 2.148e-43 | [-0.885, -0.801] | validation | 176.7 |  |
| trained_seed43 | 16 | attention | **-0.956** | 2.049e-43 | [-0.997, -0.915] | validation | 182.2 |  |
| trained_seed41 | 16 | mlp | **-0.949** | 2.049e-43 | [-0.995, -0.907] | validation | 196.1 |  |
| trained_seed42 | 16 | mlp | **-1.127** | 2.049e-43 | [-1.169, -1.086] | validation | 197.8 |  |
| trained_seed43 | 16 | mlp | **-1.064** | 2.049e-43 | [-1.105, -1.026] | validation | 204.7 |  |

## Domain: legal

| Cell | r | scheme | ΔNLL | p | 95% CI | split | latency(ms) | guard |
|---|---|---|---|---|---|---|---|---|
| trained_seed41 | 4 | all | **-1.306** | 4.056e-169 | [-1.317, -1.296] | validation | 280.9 | ⚠ |
| trained_seed42 | 4 | all | **-1.306** | 4.056e-169 | [-1.317, -1.296] | validation | 213.4 |  |
| trained_seed43 | 4 | all | **-1.306** | 4.056e-169 | [-1.317, -1.296] | validation | 221.7 |  |
| trained_seed41 | 4 | attention | **-1.206** | 4.056e-169 | [-1.216, -1.196] | validation | 183.5 |  |
| trained_seed42 | 4 | attention | **-1.205** | 4.056e-169 | [-1.215, -1.195] | validation | 233.1 |  |
| trained_seed43 | 4 | attention | **-1.208** | 4.056e-169 | [-1.218, -1.198] | validation | 176.4 |  |
| trained_seed41 | 4 | mlp | **-1.297** | 4.056e-169 | [-1.307, -1.287] | validation | 175.4 |  |
| trained_seed42 | 4 | mlp | **-1.295** | 4.056e-169 | [-1.306, -1.286] | validation | 215.7 |  |
| trained_seed43 | 4 | mlp | **-1.296** | 4.056e-169 | [-1.307, -1.286] | validation | 154.2 |  |
| trained_seed41 | 8 | all | **-1.316** | 4.056e-169 | [-1.327, -1.306] | validation | 236.2 |  |
| trained_seed42 | 8 | all | **-1.317** | 4.056e-169 | [-1.328, -1.307] | validation | 244.0 |  |
| trained_seed43 | 8 | all | **-1.317** | 4.056e-169 | [-1.328, -1.307] | validation | 327.6 | ⚠ |
| trained_seed41 | 8 | attention | **-1.232** | 4.056e-169 | [-1.243, -1.222] | validation | 176.7 |  |
| trained_seed42 | 8 | attention | **-1.230** | 4.056e-169 | [-1.241, -1.220] | validation | 172.3 |  |
| trained_seed43 | 8 | attention | **-1.229** | 4.056e-169 | [-1.240, -1.219] | validation | 186.3 |  |
| trained_seed41 | 8 | mlp | **-1.309** | 4.056e-169 | [-1.320, -1.299] | validation | 174.8 |  |
| trained_seed42 | 8 | mlp | **-1.309** | 4.056e-169 | [-1.320, -1.299] | validation | 161.4 |  |
| trained_seed43 | 8 | mlp | **-1.309** | 4.056e-169 | [-1.320, -1.299] | validation | 161.2 |  |
| trained_seed41 | 16 | all | **-1.326** | 4.056e-169 | [-1.337, -1.315] | validation | 288.6 | ⚠ |
| trained_seed42 | 16 | all | **-1.327** | 4.056e-169 | [-1.337, -1.316] | validation | 258.5 | ⚠ |
| trained_seed43 | 16 | all | **-1.325** | 4.056e-169 | [-1.336, -1.315] | validation | 268.6 | ⚠ |
| trained_seed41 | 16 | attention | **-1.255** | 4.056e-169 | [-1.265, -1.245] | validation | 162.7 |  |
| trained_seed42 | 16 | attention | **-1.254** | 4.056e-169 | [-1.265, -1.244] | validation | 182.4 |  |
| trained_seed43 | 16 | attention | **-1.252** | 4.056e-169 | [-1.263, -1.242] | validation | 171.3 |  |
| trained_seed41 | 16 | mlp | **-1.319** | 4.056e-169 | [-1.329, -1.308] | validation | 185.3 |  |
| trained_seed42 | 16 | mlp | **-1.321** | 4.056e-169 | [-1.331, -1.310] | validation | 183.7 |  |
| trained_seed43 | 16 | mlp | **-1.320** | 4.056e-169 | [-1.331, -1.310] | validation | 190.1 |  |

## Domain: medical

| Cell | r | scheme | ΔNLL | p | 95% CI | split | latency(ms) | guard |
|---|---|---|---|---|---|---|---|---|
| trained_seed41 | 4 | all | **-0.243** | 3.315e-36 | [-0.291, -0.200] | validation | 242.5 |  |
| trained_seed42 | 4 | all | **-0.243** | 3.315e-36 | [-0.291, -0.200] | validation | 246.2 |  |
| trained_seed43 | 4 | all | **-0.243** | 3.315e-36 | [-0.291, -0.200] | validation | 231.0 |  |
| trained_seed41 | 4 | attention | **-0.943** | 2.327e-88 | [-0.995, -0.899] | validation | 156.9 |  |
| trained_seed42 | 4 | attention | **-0.998** | 2.327e-88 | [-1.049, -0.952] | validation | 152.6 |  |
| trained_seed43 | 4 | attention | **-1.053** | 2.327e-88 | [-1.104, -1.009] | validation | 152.6 |  |
| trained_seed41 | 4 | mlp | **-1.233** | 2.327e-88 | [-1.288, -1.186] | validation | 144.1 |  |
| trained_seed42 | 4 | mlp | **-1.221** | 2.327e-88 | [-1.275, -1.175] | validation | 144.1 |  |
| trained_seed43 | 4 | mlp | **-1.219** | 2.327e-88 | [-1.271, -1.174] | validation | 142.4 |  |
| trained_seed41 | 8 | all | **-0.593** | 7.114e-88 | [-0.634, -0.555] | validation | 226.6 |  |
| trained_seed42 | 8 | all | **-0.870** | 2.422e-88 | [-0.916, -0.828] | validation | 222.4 |  |
| trained_seed43 | 8 | all | **-0.522** | 5.747e-85 | [-0.569, -0.480] | validation | 225.4 |  |
| trained_seed41 | 8 | attention | **-1.019** | 2.327e-88 | [-1.067, -0.977] | validation | 157.2 |  |
| trained_seed42 | 8 | attention | **-0.898** | 2.341e-88 | [-0.947, -0.858] | validation | 154.9 |  |
| trained_seed43 | 8 | attention | **-1.027** | 2.327e-88 | [-1.074, -0.985] | validation | 156.8 |  |
| trained_seed41 | 8 | mlp | **-1.242** | 2.327e-88 | [-1.295, -1.196] | validation | 151.9 |  |
| trained_seed42 | 8 | mlp | **-1.246** | 2.327e-88 | [-1.301, -1.199] | validation | 153.3 |  |
| trained_seed43 | 8 | mlp | **-1.222** | 2.327e-88 | [-1.260, -1.184] | validation | 149.5 |  |
| trained_seed41 | 16 | all | **-0.744** | 2.534e-88 | [-0.783, -0.707] | validation | 253.5 | ⚠ |
| trained_seed42 | 16 | all | **-0.697** | 2.422e-88 | [-0.752, -0.652] | validation | 246.3 |  |
| trained_seed43 | 16 | all | **-0.878** | 2.341e-88 | [-0.914, -0.842] | validation | 252.6 | ⚠ |
| trained_seed41 | 16 | attention | **-0.678** | 3.022e-88 | [-0.718, -0.642] | validation | 161.9 |  |
| trained_seed42 | 16 | attention | **-0.343** | 1.628e-53 | [-0.393, -0.297] | validation | 161.3 |  |
| trained_seed43 | 16 | attention | **-0.855** | 2.367e-88 | [-0.900, -0.816] | validation | 178.6 |  |
| trained_seed41 | 16 | mlp | **-1.256** | 2.327e-88 | [-1.310, -1.208] | validation | 188.4 |  |
| trained_seed42 | 16 | mlp | **-1.211** | 2.327e-88 | [-1.252, -1.172] | validation | 179.4 |  |
| trained_seed43 | 16 | mlp | **-1.252** | 2.327e-88 | [-1.306, -1.204] | validation | 186.0 |  |
